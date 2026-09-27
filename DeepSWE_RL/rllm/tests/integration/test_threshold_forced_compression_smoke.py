"""Real Qwen3/SWE smoke for threshold-triggered forced compression (no PPO)."""

import asyncio
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from rllm.agents.swe_agent import SWEAgent, parse_xml_response
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.swe.swe_arsenal import SWEEnv


MODEL = "/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B"
DATA = Path("rllm/data/datasets/SWE_SMITH_FULL_NONEMPTY_VERIFIED/train_verl.parquet")
REPORT = Path("/tmp/threshold_forced_compression_smoke.json")
RESPONSES = Path("/tmp/threshold_forced_compression_responses.jsonl")


class ObservedAgent(SWEAgent):
    compression_events = None

    def reset(self):
        super().reset()
        self.compression_events = []

    def apply_context_compression(self, summary):
        before = [dict(m) for m in self.messages]
        super().apply_context_compression(summary)
        self.compression_events.append({
            "summary": summary,
            "messages_before": before,
            "messages_after_apply": [dict(m) for m in self.messages],
        })
        # Smoke-only: force exactly once, then verify ordinary real-model routing.
        self.compression_force_tokens = 10**9
        self.compression_warning_tokens = 10**9 - 1


class ObservedEngine(AgentExecutionEngine):
    async def get_model_response(self, prompt, application_id, **kwargs):
        response = await super().get_model_response(prompt, application_id, **kwargs)
        record = {"response": response, "prompt": prompt, "max_tokens": kwargs.get("max_tokens")}
        try:
            _, action = parse_xml_response(response)
            record["function_name"] = action.function_name
            record["parameters"] = action.parameters
        except Exception as exc:
            record["parse_error"] = repr(exc)
        with RESPONSES.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return response


def prompt_tokens(engine, messages):
    text = engine.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
    return len(engine.tokenizer.encode(text, add_special_tokens=False))


async def main():
    REPORT.unlink(missing_ok=True)
    RESPONSES.unlink(missing_ok=True)
    entry = json.loads(pd.read_parquet(DATA).iloc[0]["extra_info"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    env = SWEEnv.from_dict({**entry, "backend": "docker", "scaffold": "sweagent", "delete_image": False})
    agent = ObservedAgent(
        scaffold="sweagent",
        enable_context_compression=True,
        context_compression_mode="forced",
        compression_warning_tokens=0,
        compression_force_tokens=1,
        compression_summary_max_tokens=1024,
        compression_retry_num=2,
    )
    env_calls = {"count": 0}
    original_step = env.step

    def counted_step(action):
        env_calls["count"] += 1
        return original_step(action)

    env.step = counted_step
    engine = ObservedEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        rollout_engine_args={"base_url": "http://127.0.0.1:8123/v1", "api_key": "local"},
        sampling_params={"model": "qwen3-4b-stage9a", "temperature": 0.3},
        n_parallel_agents=1,
        max_steps=2,
        # Short rollout budget; Stage10B exact-51200 accounting is covered by
        # CPU regression and this smoke exercises routing/context semantics.
        max_response_length=12000,
        max_prompt_length=8192,
        trajectory_timeout=7200,
        env_response_timeout=3600,
        env_response_retry_num=3,
        enforce_max_prompt_length=False,
    )
    engine.update_envs_and_agents([env], [agent])
    result = await engine.run_agent_trajectory_async(0, f"threshold-{entry['instance_id']}", mode="Token")
    segments = result["policy_segments"]
    events = []
    for event in agent.compression_events:
        events.append({
            "summary_tokens": len(tokenizer.encode(event["summary"], add_special_tokens=False)),
            "working_prompt_before": prompt_tokens(engine, event["messages_before"]),
            "working_prompt_after_apply": prompt_tokens(engine, event["messages_after_apply"]),
        })
    report = {
        "task_id": entry["instance_id"],
        "termination_reason": result["termination_reason"],
        "reward": result["trajectory_reward"],
        "trajectory_steps": len(agent.trajectory.steps),
        "env_step_count": env_calls["count"],
        "forced_calls": result["metrics"]["forced_compression_calls"],
        "autonomous_calls": result["metrics"]["autonomous_compression_calls"],
        "retry_count": result["metrics"]["forced_compression_retry_count"],
        "events": events,
        "segment_count": len(segments),
        "segments": [{
            "segment_id": s["segment_id"],
            "compression_source": s["compression_source"],
            "ended_by_compression": s["ended_by_compression"],
            "policy_tokens": sum(s["policy_mask"]),
            "observation_tokens": len(s["policy_mask"]) - sum(s["policy_mask"]),
        } for s in segments],
        "response_tokens": len(result["response_tokens"]),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    assert report["forced_calls"] == 1
    assert segments[0]["compression_source"] == "forced_threshold"
    assert segments[0]["ended_by_compression"]
    assert env_calls["count"] >= 1


if __name__ == "__main__":
    asyncio.run(main())
