import asyncio
import json
import math
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from rllm.agents.swe_agent import SWEAgent, parse_xml_response
from rllm.agents.utils import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.swe.swe_arsenal import SWEEnv


MODEL_PATH = "/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B"
DATA_PATH = Path(
    "/home/yyk/yyk11/zhongtianyang/memory/SWE-Master/DeepSWE_RL/rllm/"
    "rllm/data/datasets/SWE_SMITH_FULL_NONEMPTY_VERIFIED/train_verl.parquet"
)
OUTPUT_PATH = Path("/tmp/stage9a_swe_single_rollout_result.json")
MODEL_RESPONSE_PATH = Path("/tmp/stage9a_model_responses.jsonl")
FORCED_SUMMARY = (
    "The task is to modify the target repository. Preserve the original task "
    "requirements. So far I inspected the repository and executed the previous "
    "command. Continue from the current repository state and inspect the relevant "
    "files before editing."
)
FORCED_COMPRESSION_RESPONSE = (
    "<function=compress_context>\n"
    "<parameter=summary>\n"
    f"{FORCED_SUMMARY}\n"
    "</parameter>\n"
    "</function>"
)


class DiagnosticSWEAgent(SWEAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compression_events = []
        self.diagnostic_tokenizer = None
        self.diagnostic_parser = None
        self.diagnostic_cumulative_response_tokens = 0

    def apply_context_compression(self, summary: str) -> None:
        event = {
            "summary": summary,
            "messages_before": [dict(message) for message in self.messages],
            "trajectory_steps_before": len(self.trajectory.steps),
            "cumulative_response_tokens_before": (
                self.diagnostic_cumulative_response_tokens
            ),
        }
        super().apply_context_compression(summary)
        event["messages_after_apply"] = [dict(message) for message in self.messages]
        self.compression_events.append(event)

    def update_from_env(self, observation, reward, done, info):
        super().update_from_env(observation, reward, done, info)
        if self.trajectory.steps and self.diagnostic_tokenizer is not None:
            assistant_message, env_messages = get_recent_assistant_user_messages(
                self.chat_completions
            )
            assistant_tokens, _ = convert_messages_to_tokens_and_masks(
                [assistant_message],
                tokenizer=self.diagnostic_tokenizer,
                parser=self.diagnostic_parser,
                contains_first_msg=False,
                contains_generation_msg=False,
            )
            env_tokens, _ = convert_messages_to_tokens_and_masks(
                env_messages,
                tokenizer=self.diagnostic_tokenizer,
                parser=self.diagnostic_parser,
                contains_first_msg=False,
                contains_generation_msg=True,
            )
            self.diagnostic_cumulative_response_tokens += (
                len(assistant_tokens) + len(env_tokens)
            )
        if info.get("internal_action") == "compress_context":
            self.compression_events[-1]["messages_after_ack"] = [
                dict(message) for message in self.messages
            ]
            self.compression_events[-1]["cumulative_response_tokens_after"] = (
                self.diagnostic_cumulative_response_tokens
            )


class DiagnosticAgentExecutionEngine(AgentExecutionEngine):
    model_call_count = 0
    diagnostic_env_counter = None

    async def get_model_response(self, prompt, application_id, **kwargs):
        call_index = self.model_call_count
        self.model_call_count += 1
        if call_index == 1:
            response = FORCED_COMPRESSION_RESPONSE
            source = "forced_compression_fixture"
        else:
            response = await super().get_model_response(
                prompt, application_id, **kwargs
            )
            source = "real_qwen3"
        record = {
            "step_index": call_index,
            "source": source,
            "raw_model_response": response,
            "prompt_token_count": rendered_token_count(self, prompt),
            "env_step_count_at_response": self.diagnostic_env_counter["count"],
        }
        try:
            _, parsed_action = parse_xml_response(response)
            record.update(
                {
                    "parsed_action": parsed_action.to_xml_string(),
                    "parsed_function_name": parsed_action.function_name,
                    "parsed_parameters": parsed_action.parameters,
                }
            )
        except Exception as error:
            record["parse_error"] = repr(error)
        with MODEL_RESPONSE_PATH.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
        return response


def rendered_token_count(engine, messages):
    rendered = engine.chat_parser.parse(
        messages, add_generation_prompt=True, is_first_msg=True
    )
    return len(engine.tokenizer.encode(rendered, add_special_tokens=False))


async def main():
    MODEL_RESPONSE_PATH.unlink(missing_ok=True)
    row = pd.read_parquet(DATA_PATH).iloc[0]
    entry = json.loads(row["extra_info"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    env = SWEEnv.from_dict(
        {
            **entry,
            "backend": "docker",
            "scaffold": "sweagent",
            "delete_image": False,
        }
    )
    agent = DiagnosticSWEAgent(
        scaffold="sweagent", enable_context_compression=True
    )
    env_step_counter = {"count": 0}
    production_env_step = env.step

    def counted_env_step(action):
        env_step_counter["count"] += 1
        return production_env_step(action)

    env.step = counted_env_step
    engine = DiagnosticAgentExecutionEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        rollout_engine_args={
            "base_url": "http://127.0.0.1:8123/v1",
            "api_key": "stage9a-local",
        },
        sampling_params={
            "model": "qwen3-4b-stage9a",
            "temperature": 0.3,
        },
        n_parallel_agents=1,
        max_steps=3,
        max_response_length=51200,
        max_prompt_length=8192,
        trajectory_timeout=7200,
        env_response_timeout=3600,
        env_response_retry_num=3,
        agent_class=DiagnosticSWEAgent,
        env_class=SWEEnv,
        agent_args={"scaffold": "sweagent", "enable_context_compression": True},
        env_args={"backend": "docker", "scaffold": "sweagent", "delete_image": False},
        enforce_max_prompt_length=False,
    )
    engine.diagnostic_env_counter = env_step_counter
    agent.diagnostic_tokenizer = tokenizer
    agent.diagnostic_parser = engine.chat_parser
    engine.update_envs_and_agents([env], [agent])

    prompt_marker = "compress_context"
    marker_pos = agent.system_prompt.find(prompt_marker)
    snippet_start = max(0, marker_pos - 220)
    snippet_end = min(len(agent.system_prompt), marker_pos + 650)
    print("STAGE9A_SYSTEM_PROMPT_SNIPPET_BEGIN", flush=True)
    print(agent.system_prompt[snippet_start:snippet_end], flush=True)
    print("STAGE9A_SYSTEM_PROMPT_SNIPPET_END", flush=True)
    assert marker_pos >= 0

    result = await engine.run_agent_trajectory_async(
        idx=0,
        application_id=f"stage9a-{entry['instance_id']}",
        seed=0,
        mode="Token",
    )

    segments = result.get("policy_segments", [])
    event_reports = []
    for event in agent.compression_events:
        before = event["messages_before"]
        after_apply = event["messages_after_apply"]
        after_ack = event.get("messages_after_ack", [])
        event_reports.append(
            {
                "summary": event["summary"],
                "summary_token_count": len(
                    tokenizer.encode(event["summary"], add_special_tokens=False)
                ),
                "trajectory_steps_before": event["trajectory_steps_before"],
                "messages_before_count": len(before),
                "messages_after_apply_count": len(after_apply),
                "messages_after_ack_count": len(after_ack),
                "working_prompt_tokens_before": rendered_token_count(engine, before),
                "working_prompt_tokens_after_apply": rendered_token_count(
                    engine, after_apply
                ),
                "working_prompt_tokens_after_ack": rendered_token_count(
                    engine, after_ack
                ) if after_ack else None,
                "roles_after_ack": [message["role"] for message in after_ack],
                "raw_model_response": agent.trajectory.steps[
                    event["trajectory_steps_before"] - 1
                ].model_response,
                "cumulative_response_tokens_before": event[
                    "cumulative_response_tokens_before"
                ],
                "cumulative_response_tokens_after": event.get(
                    "cumulative_response_tokens_after"
                ),
            }
        )

    steps = agent.trajectory.steps
    segment_reports = []
    for segment in segments:
        mask = segment["policy_mask"]
        response_tokens = segment["response_tokens"]
        assert len(mask) == len(response_tokens)
        segment_reports.append(
            {
                "episode_id": segment["episode_id"],
                "segment_id": segment["segment_id"],
                "prompt_tokens": len(segment["prompt_tokens"]),
                "response_tokens": len(response_tokens),
                "policy_tokens": int(sum(mask)),
                "observation_tokens": int(len(mask) - sum(mask)),
                "ended_by_compression": segment["ended_by_compression"],
                "compression_index": segment["compression_index"],
                "prompt_contains_internal_ack": (
                    "Context compression completed." in segment["prompt_text"]
                ),
                "prompt_contains_forced_summary": (
                    FORCED_SUMMARY in segment["prompt_text"]
                ),
                "response_contains_internal_ack": (
                    "Context compression completed."
                    in tokenizer.decode(response_tokens, skip_special_tokens=False)
                ),
                "response_contains_forced_summary": (
                    FORCED_SUMMARY
                    in tokenizer.decode(response_tokens, skip_special_tokens=False)
                ),
            }
        )

    final_prompt_tokens = rendered_token_count(engine, agent.chat_completions)
    model_call_records = [
        json.loads(line)
        for line in MODEL_RESPONSE_PATH.read_text(encoding="utf-8").splitlines()
    ]
    report = {
        "task_id": entry["instance_id"],
        "docker_image": entry.get("docker_image"),
        "rollout_n": 1,
        "max_response_length": 51200,
        "temperature": 0.3,
        "compression_enabled": agent.enable_context_compression,
        "system_prompt_has_compress_context": marker_pos >= 0,
        "rollout_steps": len(steps),
        "compression_call_count": len(agent.compression_events),
        "env_step_count": env_step_counter["count"],
        "compression_events": event_reports,
        "model_calls": model_call_records,
        "segment_count": len(segments),
        "segments": segment_reports,
        "segment_ids_contiguous": [s["segment_id"] for s in segments]
        == list(range(len(segments))),
        "segment_episode_ids_unique": sorted(
            {s["episode_id"] for s in segments}
        ),
        "trajectory_steps_have_model_response": all(
            step.model_response is not None for step in steps
        ),
        "trajectory_model_responses": [step.model_response for step in steps],
        "trajectory_actions": [step.action for step in steps],
        "trajectory_observations": [step.next_observation for step in steps],
        "termination_reason": result["termination_reason"],
        "reward": float(result["trajectory_reward"]),
        "reward_is_finite": math.isfinite(float(result["trajectory_reward"])),
        "cumulative_response_tokens": result["metrics"][
            "total_response_tokens_num"
        ],
        "model_generated_tokens": result["metrics"][
            "response_tokens_model_gen_num"
        ],
        "flattened_environment_tokens": result["metrics"][
            "response_tokens_env_num"
        ],
        "segment_observation_tokens": sum(
            segment["observation_tokens"] for segment in segment_reports
        ),
        "current_working_prompt_tokens": final_prompt_tokens,
        "final_working_message_roles": [
            message["role"] for message in agent.chat_completions
        ],
    }
    validation_failures = []
    if len(model_call_records) != 3:
        validation_failures.append("expected exactly three model calls")
    if model_call_records[0].get("parsed_function_name") == "compress_context":
        validation_failures.append("first real Qwen action unexpectedly compressed")
    if not model_call_records[0].get("parsed_function_name"):
        validation_failures.append("first real Qwen action was empty")
    if model_call_records[1].get("parsed_function_name") != "compress_context":
        validation_failures.append("forced response did not parse as compress_context")
    if model_call_records[1].get("parsed_parameters") != {"summary": FORCED_SUMMARY}:
        validation_failures.append("forced summary changed during production parsing")
    if model_call_records[1]["env_step_count_at_response"] != 1:
        validation_failures.append("expected one env.step before compression")
    if model_call_records[2]["env_step_count_at_response"] != 1:
        validation_failures.append("compression unexpectedly called env.step")
    if model_call_records[2].get("parsed_function_name") in (None, "", "compress_context"):
        validation_failures.append("post-compression Qwen action was not an ordinary environment action")
    if env_step_counter["count"] != 2:
        validation_failures.append("post-compression ordinary action did not restore env routing")
    if len(agent.compression_events) != 1:
        validation_failures.append("expected exactly one compression event")
    if len(segments) != 2:
        validation_failures.append("expected exactly two policy segments")
    if event_reports and event_reports[0]["roles_after_ack"] != [
        "system", "user", "assistant", "tool"
    ]:
        validation_failures.append("compressed working context roles are incorrect")
    if len(segments) == 2:
        if not segment_reports[0]["response_contains_forced_summary"]:
            validation_failures.append("summary missing from segment 0 policy response")
        if segment_reports[0]["response_contains_internal_ack"]:
            validation_failures.append("internal acknowledgement leaked into segment 0 response")
        if not segment_reports[1]["prompt_contains_internal_ack"]:
            validation_failures.append("internal acknowledgement missing from segment 1 prompt")
        if not segment_reports[1]["prompt_contains_forced_summary"]:
            validation_failures.append("canonical summary missing from segment 1 prompt")
    if not all(step.model_response is not None for step in steps):
        validation_failures.append("full trajectory lost a model response")
    report["validation_failures"] = validation_failures
    report["stage9b_passed"] = not validation_failures
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("STAGE9A_RESULT_BEGIN", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("STAGE9A_RESULT_END", flush=True)
    if validation_failures:
        raise AssertionError("; ".join(validation_failures))


if __name__ == "__main__":
    asyncio.run(main())
