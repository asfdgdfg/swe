"""Real Qwen3 tokenizer/SWE smoke for an infeasible forced compression."""

import asyncio
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from rllm.agents.swe_agent import SWEAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.swe.swe_arsenal import SWEEnv


MODEL = "/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B"
DATA = Path("rllm/data/datasets/SWE_SMITH_FULL_NONEMPTY_VERIFIED/train_verl.parquet")
REPORT = Path("/tmp/threshold_forced_compression_insufficient_budget_smoke.json")


class NoModelCallEngine(AgentExecutionEngine):
    model_calls = 0

    async def get_model_response(self, prompt, application_id, **kwargs):
        self.model_calls += 1
        raise AssertionError("infeasible forced compression must not call the model")


async def main():
    REPORT.unlink(missing_ok=True)
    entry = json.loads(pd.read_parquet(DATA).iloc[0]["extra_info"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    env = SWEEnv.from_dict(
        {**entry, "backend": "docker", "scaffold": "sweagent", "delete_image": False}
    )
    agent = SWEAgent(
        scaffold="sweagent",
        enable_context_compression=True,
        context_compression_mode="forced",
        compression_warning_tokens=0,
        compression_force_tokens=1,
        compression_summary_max_tokens=1024,
        compression_retry_num=2,
        compression_force_min_response_budget=2048,
    )
    env_calls = {"count": 0}
    original_step = env.step

    def counted_step(action):
        env_calls["count"] += 1
        return original_step(action)

    env.step = counted_step
    engine = NoModelCallEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        rollout_engine_args={"base_url": "http://127.0.0.1:8123/v1", "api_key": "local"},
        sampling_params={"model": "qwen3-4b-budget-smoke", "temperature": 0.3},
        n_parallel_agents=1,
        max_steps=2,
        max_response_length=114,
        max_prompt_length=8192,
        trajectory_timeout=7200,
        env_response_timeout=3600,
        env_response_retry_num=3,
        enforce_max_prompt_length=False,
    )
    engine.update_envs_and_agents([env], [agent])
    result = await engine.run_agent_trajectory_async(
        0, f"insufficient-budget-{entry['instance_id']}", mode="Token"
    )
    candidate = result["forced_compression_candidates"][0]
    report = {
        "task_id": entry["instance_id"],
        "termination_reason": result["termination_reason"],
        "model_calls": engine.model_calls,
        "retry_count": result["metrics"]["forced_compression_retry_count"],
        "env_step_count": env_calls["count"],
        "forced_compression_failed": result["metrics"]["forced_compression_failed"],
        "forced_compression_skipped_insufficient_budget": result["metrics"][
            "forced_compression_skipped_insufficient_budget"
        ],
        "candidate": candidate,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    assert report["termination_reason"] == "TRUNCATION"
    assert report["model_calls"] == 0
    assert report["retry_count"] == 0
    assert report["env_step_count"] == 0
    assert report["forced_compression_failed"] == 0
    assert report["forced_compression_skipped_insufficient_budget"] == 1
    assert candidate["remaining_response_budget"] == 114
    assert candidate["force_min_response_budget"] == 2048


if __name__ == "__main__":
    asyncio.run(main())
