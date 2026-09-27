import asyncio
import concurrent.futures
from types import MethodType
from unittest.mock import MagicMock

from rllm.agents.swe_agent import SWEAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.agents.system_prompts import (
    SWE_CONTEXT_COMPRESSION_REQUIRED_PROMPT,
    SWE_CONTEXT_COMPRESSION_WARNING_PROMPT,
)


class FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 1 for c in text]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(int(t) - 1) for t in tokens)


class FakeParser:
    assistant_token = "<assistant>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if len(messages) == 1 and messages[0]["role"] == "assistant":
            text = "<assistant>" + messages[0]["content"]
        return text


class FakeEnv:
    def __init__(self, results=()):
        self.entry = {"instance_id": "threshold-compression"}
        self.idx = 0
        self.has_container = True
        self.step = MagicMock(side_effect=list(results))
        self.close = MagicMock()

    def reset(self):
        return "Fix the threshold compression issue.", {}

    def compute_final_reward(self):
        return 0.0


def response(function="execute_bash", parameter="command", value="pwd", thought="THOUGHT"):
    return (
        f"{thought}\n<function={function}>\n"
        f"<parameter={parameter}>{value}</parameter>\n</function>"
    )


def compression(summary, thought="COMPRESSION_THOUGHT"):
    return response("compress_context", "summary", summary, thought)


def run(agent, responses, env_results=(), max_steps=5, max_response_length=100_000):
    engine = object.__new__(AgentExecutionEngine)
    engine.agents = [agent]
    engine.envs = [FakeEnv(env_results)]
    engine.max_steps = max_steps
    engine.max_response_length = max_response_length
    engine.max_prompt_length = 100_000
    engine.enforce_max_prompt_length = False
    engine.trajectory_timeout = 1000
    engine.env_response_timeout = 10
    engine.env_response_retry_num = 2
    engine.overlong_filter = False
    engine.gamma = 0.2
    engine.tokenizer = FakeTokenizer()
    engine.chat_parser = FakeParser()
    engine.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    prompts = []
    iterator = iter(responses)

    async def fake_model(self, prompt, application_id, **kwargs):
        prompts.append([dict(m) for m in prompt])
        return next(iterator)

    engine.get_model_response = MethodType(fake_model, engine)
    try:
        result = asyncio.run(engine.run_agent_trajectory_async(0, "episode-threshold", mode="Token"))
    finally:
        engine.executor.shutdown(wait=True)
    return result, engine.envs[0], prompts


def test_below_warning_uses_unmodified_prompt_and_normal_routing():
    agent = SWEAgent(
        enable_context_compression=True,
        context_compression_mode="hybrid",
        compression_warning_tokens=90_000,
        compression_force_tokens=95_000,
    )
    result, env, prompts = run(agent, [response()], [("done", 0.0, True, {})])
    assert SWE_CONTEXT_COMPRESSION_WARNING_PROMPT not in str(prompts[0])
    assert SWE_CONTEXT_COMPRESSION_REQUIRED_PROMPT not in str(prompts[0])
    assert env.step.call_count == 1
    assert result["metrics"]["forced_compression_calls"] == 0


def test_warning_is_temporary_and_does_not_block_normal_action():
    agent = SWEAgent(
        enable_context_compression=True,
        context_compression_mode="hybrid",
        compression_warning_tokens=1,
        compression_force_tokens=95_000,
    )
    result, env, prompts = run(agent, [response()], [("done", 0.0, True, {})])
    assert SWE_CONTEXT_COMPRESSION_WARNING_PROMPT in str(prompts[0])
    assert SWE_CONTEXT_COMPRESSION_WARNING_PROMPT not in str(agent.chat_completions)
    assert env.step.call_count == 1
    assert result["termination_reason"] == "ENV_DONE"


def test_forced_retry_is_not_committed_and_valid_response_is_full_policy():
    summary = "goal state tests files next action"
    agent = SWEAgent(
        enable_context_compression=True,
        context_compression_mode="hybrid",
        compression_warning_tokens=7_500,
        compression_force_tokens=8_000,
        compression_retry_num=2,
        compression_summary_max_tokens=100,
    )
    before_threshold = response(value="echo grow", thought="BEFORE_THRESHOLD")
    invalid = response(value="echo invalid", thought="FAILED_RETRY")
    valid = compression(summary, thought="VALID_FORCED_THOUGHT")
    normal = response(value="pwd", thought="AFTER_COMPRESSION")
    result, env, prompts = run(
        agent,
        [before_threshold, invalid, valid, normal],
        [("O" * 2_000, 0.0, False, {}), ("terminal", 0.0, True, {})],
    )

    assert env.step.call_count == 2
    assert len(result["forced_compression_retries"]) == 1
    assert "FAILED_RETRY" in result["forced_compression_retries"][0]["response"]
    assert all("FAILED_RETRY" not in step.model_response for step in agent.trajectory.steps)
    assert len(result["policy_segments"]) == 2
    first = result["policy_segments"][0]
    assert first["compression_source"] == "forced_threshold"
    assert first["ended_by_compression"]
    first_text = FakeTokenizer().decode(first["response_tokens"])
    assert "VALID_FORCED_THOUGHT" in first_text
    assert "<function=compress_context>" in first_text
    assert summary in first_text
    valid_tokens = FakeTokenizer().encode(valid)
    valid_start = next(
        i for i in range(len(first["response_tokens"]) - len(valid_tokens) + 1)
        if first["response_tokens"][i : i + len(valid_tokens)] == valid_tokens
    )
    assert first["policy_mask"][valid_start : valid_start + len(valid_tokens)] == [1] * len(valid_tokens)
    assert result["metrics"]["forced_compression_calls"] == 1
    assert result["metrics"]["autonomous_compression_calls"] == 0
    def has_required(prompt):
        return any(m["content"] == SWE_CONTEXT_COMPRESSION_REQUIRED_PROMPT for m in prompt)

    assert not has_required(prompts[0])
    assert has_required(prompts[1])
    assert has_required(prompts[2])
    assert not has_required(prompts[3])


def test_forced_summary_limit_retries_then_terminates_without_env_or_policy():
    agent = SWEAgent(
        enable_context_compression=True,
        context_compression_mode="forced",
        compression_warning_tokens=0,
        compression_force_tokens=1,
        compression_retry_num=1,
        compression_summary_max_tokens=3,
    )
    too_long = compression("summary is too long")
    result, env, _ = run(agent, [too_long, too_long], [])
    assert result["termination_reason"] == "FORCED_COMPRESSION_FAILED"
    assert env.step.call_count == 0
    assert len(agent.trajectory.steps) == 0
    assert result["policy_segments"] == []
    assert len(result["forced_compression_retries"]) == 2
    assert result["metrics"]["forced_compression_failed"] == 1
    assert result["metrics"]["forced_compression_skipped_insufficient_budget"] == 0


def test_force_at_minimum_response_budget_calls_model():
    agent = SWEAgent(
        enable_context_compression=True,
        context_compression_mode="forced",
        compression_warning_tokens=0,
        compression_force_tokens=1,
        compression_force_min_response_budget=2048,
    )
    result, env, prompts = run(
        agent,
        [compression("goal state files next action")],
        [],
        max_steps=1,
        max_response_length=2048,
    )
    assert len(prompts) == 1
    assert env.step.call_count == 0
    assert result["metrics"]["forced_compression_calls"] == 1
    assert result["metrics"]["forced_compression_skipped_insufficient_budget"] == 0
    assert result["forced_compression_candidates"][0]["remaining_response_budget"] == 2048
    event = result["compression_events"][0]
    assert event["source"] == "forced_threshold"
    assert event["summary_tokens"] > 0
    assert event["working_prompt_after"] > 0


def test_force_with_2047_response_tokens_remaining_truncates_without_model_call():
    agent = SWEAgent(
        enable_context_compression=True,
        context_compression_mode="forced",
        compression_warning_tokens=0,
        compression_force_tokens=1,
        compression_force_min_response_budget=2048,
    )
    result, env, prompts = run(
        agent, [], [], max_response_length=2047
    )
    assert result["termination_reason"] == "TRUNCATION"
    assert prompts == []
    assert env.step.call_count == 0
    assert result["metrics"]["forced_compression_retry_count"] == 0
    assert result["metrics"]["forced_compression_failed"] == 0
    assert result["metrics"]["forced_compression_skipped_insufficient_budget"] == 1
    candidate = result["forced_compression_candidates"][0]
    assert candidate["remaining_response_budget"] == 2047
    assert candidate["force_min_response_budget"] == 2048


def test_force_with_114_response_tokens_remaining_reproduces_pilot_as_truncation():
    agent = SWEAgent(
        enable_context_compression=True,
        context_compression_mode="forced",
        compression_warning_tokens=0,
        compression_force_tokens=1,
        compression_force_min_response_budget=2048,
    )
    result, env, prompts = run(
        agent, [], [], max_response_length=114
    )
    assert result["termination_reason"] == "TRUNCATION"
    assert prompts == []
    assert env.step.call_count == 0
    assert result["metrics"]["forced_compression_retry_count"] == 0
    assert result["metrics"]["forced_compression_failed"] == 0
    assert result["metrics"]["forced_compression_skipped_insufficient_budget"] == 1
    candidate = result["forced_compression_candidates"][0]
    assert candidate["step_idx"] == 0
    assert candidate["response_token_len"] == 0
    assert candidate["remaining_response_budget"] == 114
    assert candidate["force_min_response_budget"] == 2048
    assert candidate["status"] == "skipped_insufficient_budget"
    assert candidate["termination_reason"] == "TRUNCATION"


def test_autonomous_and_forced_identical_response_have_identical_policy_tokens():
    raw = compression("same summary")
    autonomous = SWEAgent(enable_context_compression=True, context_compression_mode="autonomous")
    forced = SWEAgent(
        enable_context_compression=True,
        context_compression_mode="forced",
        compression_warning_tokens=0,
        compression_force_tokens=1,
    )
    auto_result, _, _ = run(autonomous, [raw], [], max_steps=1)
    forced_result, _, _ = run(forced, [raw], [], max_steps=1)
    auto_segment = auto_result["policy_segments"][0]
    forced_segment = forced_result["policy_segments"][0]
    assert auto_segment["response_tokens"] == forced_segment["response_tokens"]
    assert auto_segment["policy_mask"] == forced_segment["policy_mask"]
    assert auto_segment["compression_source"] == "autonomous"
    assert forced_segment["compression_source"] == "forced_threshold"
