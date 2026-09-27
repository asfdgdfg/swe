"""Paired inference-only SWE prompt exposure experiment.

This script deliberately uses AgentExecutionEngine directly.  It never constructs a
trainer and therefore cannot compute advantages or update model parameters.
"""

import argparse
import asyncio
import hashlib
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from rllm.agents.swe_agent import SWEAgent, parse_compress_context_action, parse_xml_response
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.swe.swe_arsenal import SWEEnv


KNOWN_FUNCTIONS = {"execute_bash", "str_replace_editor", "submit", "compress_context"}
FUNCTION_RE = re.compile(r"<function=([^>\s]+)")


class RecordingEngine(AgentExecutionEngine):
    def __init__(self, *args, raw_response_path: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_response_path = raw_response_path
        self.response_records = {}
        self.episode_seeds = {}

    async def get_model_response(self, prompt, application_id, **kwargs):
        records = self.response_records.setdefault(application_id, [])
        # The base OpenAI engine currently ignores run_agent_trajectory_async's
        # seed argument.  Pin each paired model call explicitly in this
        # inference-only fixture so A/B share the same random stream while their
        # trajectories have a common prefix.
        kwargs["seed"] = self.episode_seeds[application_id] + len(records)
        response = await super().get_model_response(prompt, application_id, **kwargs)
        record = {
            "episode_id": application_id,
            "model_call_index": len(records),
            "raw_model_response": response,
        }
        try:
            _, action = parse_xml_response(response)
            record["parsed_function_name"] = action.function_name
            record["parsed_parameters"] = action.parameters
        except Exception as error:
            record["parse_error"] = repr(error)
        records.append(record)
        with self.raw_response_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        return response


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_subsequence(sequence, target):
    if not target:
        return None
    for index in range(len(sequence) - len(target) + 1):
        if sequence[index : index + len(target)] == target:
            return index
    return None


def summary_presence(summary: str) -> dict:
    lowered = summary.lower()
    return {
        "task_goal": bool(re.search(r"task|issue|goal|requirement|任务|目标|要求", lowered)),
        "completed_actions": bool(re.search(r"completed|done|implemented|changed|edited|已完成|修改|执行", lowered)),
        "important_observations": bool(re.search(r"observ|found|test|error|result|发现|测试|错误|结果", lowered)),
        "created_or_modified_files": bool(re.search(r"file|created|modified|/[\w./-]+|文件|创建", lowered)),
        "next_step": bool(re.search(r"next|continue|remaining|下一步|继续|剩余", lowered)),
    }


def analyze_episode(variant, prompt_hash, task_id, seed, result, agent, records, tokenizer):
    counts = Counter()
    malformed_xml = 0
    invalid_compression = 0
    unknown_function = 0
    summaries = []
    for record in records:
        raw = record["raw_model_response"]
        name = record.get("parsed_function_name", "")
        if "parse_error" in record or ("<function=" in raw and "</function>" not in raw):
            malformed_xml += 1
        if name:
            counts[name] += 1
            if name not in KNOWN_FUNCTIONS:
                unknown_function += 1
            if name == "compress_context" and "parse_error" not in record:
                try:
                    _, action = parse_xml_response(raw)
                    parsed = parse_compress_context_action(action)
                    if not parsed.is_valid:
                        invalid_compression += 1
                    elif parsed.summary is not None:
                        summaries.append(parsed.summary)
                except Exception:
                    invalid_compression += 1

    segments = result.get("policy_segments", [])
    marker_tokens = tokenizer.encode("<function=compress_context>", add_special_tokens=False)
    first_position = None
    cumulative = 0
    for segment in segments:
        response_tokens = list(segment["response_tokens"])
        marker_index = find_subsequence(response_tokens, marker_tokens)
        if marker_index is not None:
            first_position = cumulative + marker_index
            break
        cumulative += len(response_tokens)

    quality = [summary_presence(summary) for summary in summaries]
    metrics = result["metrics"]
    return {
        "prompt_variant": variant,
        "system_prompt_sha256": prompt_hash,
        "task_id": task_id,
        "rollout_seed": seed,
        "episode_id": result.get("episode_id", f"{variant}-{task_id}-{seed}"),
        "compression_call_count": counts["compress_context"],
        "first_compression_response_token_position": first_position,
        "summary_token_count": [len(tokenizer.encode(s, add_special_tokens=False)) for s in summaries],
        "compression_summary_presence": quality,
        "invalid_compression_schema_count": invalid_compression,
        "malformed_xml_count": malformed_xml,
        "unknown_function_count": unknown_function,
        "execute_bash_count": counts["execute_bash"],
        "str_replace_editor_count": counts["str_replace_editor"],
        "submit_count": counts["submit"],
        "reward": float(result["trajectory_reward"]),
        "reward_is_finite": math.isfinite(float(result["trajectory_reward"])),
        "termination_reason": result["termination_reason"],
        "response_tokens": int(metrics["total_response_tokens_num"]),
        "rollout_steps": len(agent.trajectory.steps),
        "segment_count": len(segments),
    }


async def run_variant(args, entries, seeds, tokenizer, variant, output_dir):
    agents = [
        SWEAgent(
            scaffold="sweagent",
            enable_context_compression=True,
            context_compression_prompt_style=variant,
        )
        for _ in entries
    ]
    prompt = agents[0].system_prompt
    assert all(agent.system_prompt == prompt for agent in agents)
    prompt_hash = sha256_text(prompt)
    (output_dir / f"system_prompt_{variant}.txt").write_text(prompt, encoding="utf-8")
    (output_dir / f"system_prompt_{variant}.sha256").write_text(prompt_hash + "\n", encoding="utf-8")

    envs = [
        SWEEnv.from_dict({**entry, "backend": "docker", "scaffold": "sweagent", "delete_image": False})
        for entry in entries
    ]
    raw_path = output_dir / f"raw_responses_{variant}.jsonl"
    raw_path.unlink(missing_ok=True)
    engine = RecordingEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        raw_response_path=raw_path,
        rollout_engine_args={"base_url": args.base_url, "api_key": "prompt-exposure-ab"},
        sampling_params={"model": args.served_model_name, "temperature": 0.3},
        n_parallel_agents=args.parallel_agents,
        max_steps=200,
        max_response_length=51200,
        max_prompt_length=8192,
        trajectory_timeout=7200,
        env_response_timeout=3600,
        env_response_retry_num=3,
        agent_class=SWEAgent,
        env_class=SWEEnv,
        agent_args={
            "scaffold": "sweagent",
            "enable_context_compression": True,
            "context_compression_prompt_style": variant,
        },
        env_args={"backend": "docker", "scaffold": "sweagent", "delete_image": False},
        enforce_max_prompt_length=False,
    )
    engine.update_envs_and_agents(envs, agents)
    engine.executor = ThreadPoolExecutor(max_workers=args.parallel_agents)
    semaphore = asyncio.Semaphore(args.parallel_agents)

    async def run_one(index):
        task_id = entries[index]["instance_id"]
        episode_id = f"ab-{variant}-{index:02d}-{task_id}"
        engine.episode_seeds[episode_id] = seeds[index]
        async with semaphore:
            result = await engine.run_agent_trajectory_with_retry(
                index, application_id=episode_id, seed=seeds[index], mode="Token"
            )
        return analyze_episode(
            variant, prompt_hash, task_id, seeds[index], result, agents[index],
            engine.response_records.get(episode_id, []), tokenizer
        )

    try:
        reports = await asyncio.gather(*(run_one(i) for i in range(len(entries))))
    finally:
        engine.executor.shutdown(wait=False, cancel_futures=True)
    path = output_dir / f"episodes_{variant}.jsonl"
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in reports), encoding="utf-8")
    return reports, prompt_hash


def aggregate(reports):
    episodes = len(reports)
    compression_episodes = sum(x["compression_call_count"] > 0 for x in reports)
    total_calls = sum(x["compression_call_count"] for x in reports)
    positions = [x["first_compression_response_token_position"] for x in reports if x["first_compression_response_token_position"] is not None]
    bins = Counter()
    for position in positions:
        bins["<5k" if position < 5000 else "5k-20k" if position < 20000 else "20k-40k" if position < 40000 else ">40k"] += 1
    return {
        "episodes": episodes,
        "episodes_with_compression": compression_episodes,
        "episodes_with_compression_rate": compression_episodes / episodes,
        "total_compression_calls": total_calls,
        "compression_calls_per_episode": total_calls / episodes,
        "first_compression_position_bins": dict(bins),
        "invalid_compression_schema_count": sum(x["invalid_compression_schema_count"] for x in reports),
        "malformed_xml_count": sum(x["malformed_xml_count"] for x in reports),
        "unknown_function_count": sum(x["unknown_function_count"] for x in reports),
        "execute_bash_count": sum(x["execute_bash_count"] for x in reports),
        "str_replace_editor_count": sum(x["str_replace_editor_count"] for x in reports),
        "submit_count": sum(x["submit_count"] for x in reports),
        "reward_mean": sum(x["reward"] for x in reports) / episodes,
        "reward_one_count": sum(x["reward"] == 1.0 for x in reports),
        "termination": dict(Counter(x["termination_reason"] for x in reports)),
        "mean_response_tokens": sum(x["response_tokens"] for x in reports) / episodes,
        "mean_rollout_steps": sum(x["rollout_steps"] for x in reports) / episodes,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8123/v1")
    parser.add_argument("--served-model-name", default="qwen3-4b-prompt-ab")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--parallel-agents", type=int, default=32)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.data_path).iloc[
        args.start_index : args.start_index + args.episodes
    ]
    if len(frame) != args.episodes:
        raise ValueError(
            f"Requested {args.episodes} episodes from start index {args.start_index}, "
            f"but only found {len(frame)} rows"
        )
    entries = [json.loads(value) for value in frame["extra_info"]]
    seeds = list(range(args.start_index, args.start_index + args.episodes))
    manifest = [{"task_id": e["instance_id"], "rollout_seed": seeds[i]} for i, e in enumerate(entries)]
    (output_dir / "paired_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    all_reports = {}
    hashes = {}
    for variant in ("additional", "inventory"):
        reports, prompt_hash = await run_variant(args, entries, seeds, tokenizer, variant, output_dir)
        all_reports[variant] = reports
        hashes[variant] = prompt_hash
    assert [x["task_id"] for x in all_reports["additional"]] == [x["task_id"] for x in all_reports["inventory"]]
    assert [x["rollout_seed"] for x in all_reports["additional"]] == [x["rollout_seed"] for x in all_reports["inventory"]]
    summary = {"system_prompt_sha256": hashes, "additional": aggregate(all_reports["additional"]), "inventory": aggregate(all_reports["inventory"])}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
