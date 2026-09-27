import asyncio
import concurrent.futures
from pathlib import Path
from types import MethodType
from unittest.mock import MagicMock

import pytest

from rllm.agents.swe_agent import SWEAgent
from rllm.agents.utils import convert_messages_to_tokens_and_masks
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.parser.chat_template.parser import QwenChatTemplateParser


ACKNOWLEDGEMENT = "Context compression completed. Continue the original task using the summary above."


class FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(int(token) - 1) for token in tokens)


class FakeChatParser:
    assistant_token = "<assistant>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False):
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += self.assistant_token
        if len(messages) == 1 and messages[0]["role"] == "assistant":
            rendered = self.assistant_token + messages[0]["content"]
        return rendered


class SuffixedFakeChatParser(FakeChatParser):
    assistant_suffix = "<eot>\n"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False):
        rendered = super().parse(messages, add_generation_prompt, is_first_msg)
        if len(messages) == 1 and messages[0]["role"] == "assistant":
            rendered += self.assistant_suffix
        return rendered


class FakeEnv:
    def __init__(self, step_results):
        self.entry = {"instance_id": "policy-segmentation-test"}
        self.idx = 0
        self.has_container = True
        self.step = MagicMock(side_effect=step_results)
        self.close = MagicMock()

    def reset(self):
        return "Fix the policy-segmentation test issue.", {}

    def compute_final_reward(self):
        return 0.0


def action_response(label, function_name="execute_bash", parameter_name="command", value="pwd"):
    return (
        f"{label}\n"
        f"<function={function_name}>\n"
        f"<parameter={parameter_name}>{value}</parameter>\n"
        "</function>"
    )


def compression_response(label, summary):
    return action_response(label, "compress_context", "summary", summary)


def make_engine(agent, env, responses, max_steps, max_response_length=100_000, chat_parser=None):
    engine = object.__new__(AgentExecutionEngine)
    engine.agents = [agent]
    engine.envs = [env]
    engine.max_steps = max_steps
    engine.max_response_length = max_response_length
    engine.max_prompt_length = 100_000
    engine.enforce_max_prompt_length = False
    engine.trajectory_timeout = 1_000
    engine.env_response_timeout = 10
    engine.env_response_retry_num = 2
    engine.overlong_filter = False
    engine.gamma = 0.2
    engine.tokenizer = FakeTokenizer()
    engine.chat_parser = chat_parser or FakeChatParser()
    engine.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    response_iter = iter(responses)
    engine.model_call_count = 0
    engine.model_max_tokens = []

    async def fake_get_model_response(self, prompt, application_id, **kwargs):
        self.model_call_count += 1
        self.model_max_tokens.append(kwargs.get("max_tokens"))
        return next(response_iter)

    engine.get_model_response = MethodType(fake_get_model_response, engine)
    return engine


def run_token_trajectory(
    agent,
    env,
    responses,
    max_steps,
    episode_id="episode-test",
    max_response_length=100_000,
    chat_parser=None,
):
    engine = make_engine(
        agent,
        env,
        responses,
        max_steps,
        max_response_length=max_response_length,
        chat_parser=chat_parser,
    )
    try:
        result = asyncio.run(
            engine.run_agent_trajectory_async(
                0,
                application_id=episode_id,
                mode="Token",
            )
        )
        return result, engine
    finally:
        engine.executor.shutdown(wait=True)


def decoded(tokens):
    return FakeTokenizer().decode(tokens)


def find_subsequence(sequence, subsequence):
    for start in range(len(sequence) - len(subsequence) + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    raise AssertionError(f"subsequence not found: {subsequence}")


def assert_segment_well_formed(segment, episode_id):
    assert segment["episode_id"] == episode_id
    assert segment["prompt_tokens"] == FakeTokenizer().encode(segment["prompt_text"])
    assert len(segment["response_tokens"]) == len(segment["policy_mask"])


def test_exact_serialized_assistant_budget_is_shared_by_flattened_and_segment_paths():
    parser = SuffixedFakeChatParser()
    response = action_response("EXACT_LIMIT")
    overhead = len(FakeTokenizer().encode(parser.assistant_suffix))
    max_response_length = len(FakeTokenizer().encode(response)) + overhead
    env = FakeEnv([("TERMINAL_OBSERVATION_MUST_NOT_ENTER_SEGMENT", 0.0, True, {})])
    agent = SWEAgent(enable_context_compression=True)

    result, engine = run_token_trajectory(
        agent,
        env,
        [response],
        max_steps=2,
        max_response_length=max_response_length,
        chat_parser=parser,
    )

    assert engine.model_max_tokens == [max_response_length - overhead]
    assert len(result["response_tokens"]) == max_response_length
    assert len(result["policy_segments"]) == 1
    segment = result["policy_segments"][0]
    assert len(segment["response_tokens"]) == max_response_length
    assert segment["response_tokens"] == result["response_tokens"].tolist()
    assert segment["policy_mask"] == [1] * max_response_length
    assert "TERMINAL_OBSERVATION_MUST_NOT_ENTER_SEGMENT" not in decoded(segment["response_tokens"])


def test_remaining_budget_not_larger_than_assistant_overhead_skips_model_call():
    parser = SuffixedFakeChatParser()
    overhead = len(FakeTokenizer().encode(parser.assistant_suffix))
    env = FakeEnv([])
    agent = SWEAgent(enable_context_compression=True)

    result, engine = run_token_trajectory(
        agent,
        env,
        [],
        max_steps=2,
        max_response_length=overhead - 1,
        chat_parser=parser,
    )

    assert engine.model_call_count == 0
    assert env.step.call_count == 0
    assert result["termination_reason"] == "TRUNCATION"
    assert result["policy_segments"] == []


def test_feature_disabled_uses_serialized_budget_without_returning_segments():
    parser = SuffixedFakeChatParser()
    response = action_response("BASELINE_EXACT_LIMIT")
    overhead = len(FakeTokenizer().encode(parser.assistant_suffix))
    max_response_length = len(FakeTokenizer().encode(response)) + overhead
    env = FakeEnv([("BASELINE_TERMINAL", 0.0, True, {})])
    agent = SWEAgent(enable_context_compression=False)

    result, engine = run_token_trajectory(
        agent,
        env,
        [response],
        max_steps=2,
        max_response_length=max_response_length,
        chat_parser=parser,
    )

    assert engine.model_max_tokens == [max_response_length - overhead]
    assert len(result["response_tokens"]) == max_response_length
    assert "policy_segments" not in result
    assert "episode_id" not in result


def test_qwen3_assistant_serialization_overhead_is_parser_derived():
    model_path = Path("/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B")
    if not model_path.exists():
        pytest.skip("local Qwen3-4B tokenizer is unavailable")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    parser = QwenChatTemplateParser(tokenizer, disable_thinking=False)
    tokens, masks = convert_messages_to_tokens_and_masks(
        [{"role": "assistant", "content": ""}],
        tokenizer=tokenizer,
        parser=parser,
        contains_first_msg=False,
        contains_generation_msg=False,
    )

    assert tokens == [151645, 198]
    assert masks == [1, 1]


def test_no_compression_has_one_segment_and_drops_unused_terminal_observation():
    responses = [action_response("A1"), action_response("A2")]
    env = FakeEnv(
        [
            ("O1_NONTERMINAL", 0.0, False, {}),
            ("O2_TERMINAL", 0.0, True, {}),
        ]
    )
    agent = SWEAgent(enable_context_compression=True)

    result, _ = run_token_trajectory(agent, env, responses, max_steps=5, episode_id="episode-no-compression")

    assert result["episode_id"] == "episode-no-compression"
    assert len(result["policy_segments"]) == 1
    segment = result["policy_segments"][0]
    assert_segment_well_formed(segment, "episode-no-compression")
    assert segment["segment_id"] == 0
    assert segment["start_model_call"] == 0
    assert segment["end_model_call"] == 1
    assert not segment["ended_by_compression"]

    response_text = decoded(segment["response_tokens"])
    assert "A1" in response_text
    assert "O1_NONTERMINAL" in response_text
    assert "A2" in response_text
    assert response_text.index("A1") < response_text.index("O1_NONTERMINAL") < response_text.index("A2")
    assert "O2_TERMINAL" not in response_text

    observation_tokens = FakeTokenizer().encode("O1_NONTERMINAL")
    observation_start = find_subsequence(segment["response_tokens"], observation_tokens)
    assert segment["policy_mask"][observation_start : observation_start + len(observation_tokens)] == [0] * len(observation_tokens)
    for response in responses:
        response_tokens = FakeTokenizer().encode(response)
        response_start = find_subsequence(segment["response_tokens"], response_tokens)
        assert segment["policy_mask"][response_start : response_start + len(response_tokens)] == [1] * len(response_tokens)
    assert agent.trajectory.steps[-1].next_observation.startswith("O2_TERMINAL")


def test_one_compression_creates_real_boundary_and_preserves_raw_response():
    summary = "S1_UNIQUE_SUMMARY"
    responses = [
        action_response("A1"),
        action_response("A2"),
        compression_response("RAW_COMPRESSION_THOUGHT", summary),
        action_response("A3"),
    ]
    env = FakeEnv(
        [
            ("O1", 0.0, False, {}),
            ("O2", 0.0, False, {}),
            ("O3_TERMINAL", 0.0, True, {}),
        ]
    )
    agent = SWEAgent(enable_context_compression=True)

    result, _ = run_token_trajectory(agent, env, responses, max_steps=8, episode_id="episode-one-compression")

    assert len(result["policy_segments"]) == 2
    first, second = result["policy_segments"]
    assert_segment_well_formed(first, "episode-one-compression")
    assert_segment_well_formed(second, "episode-one-compression")
    assert first["ended_by_compression"]
    assert first["compression_index"] == 0
    assert first["end_model_call"] == 2
    assert second["segment_id"] == 1
    assert second["start_model_call"] == 3

    first_response_text = decoded(first["response_tokens"])
    assert "A1" in first_response_text
    assert "O1" in first_response_text
    assert "A2" in first_response_text
    assert "O2" in first_response_text
    assert "RAW_COMPRESSION_THOUGHT" in first_response_text
    assert summary in first_response_text
    assert ACKNOWLEDGEMENT not in first_response_text

    summary_tokens = FakeTokenizer().encode(summary)
    summary_start = find_subsequence(first["response_tokens"], summary_tokens)
    assert first["policy_mask"][summary_start : summary_start + len(summary_tokens)] == [1] * len(summary_tokens)
    function_tokens = FakeTokenizer().encode("compress_context")
    function_start = find_subsequence(first["response_tokens"], function_tokens)
    assert first["policy_mask"][function_start : function_start + len(function_tokens)] == [1] * len(function_tokens)

    assert summary in second["prompt_text"]
    assert ACKNOWLEDGEMENT in second["prompt_text"]
    assert "A1" not in second["prompt_text"]
    assert "O1" not in second["prompt_text"]
    assert "A2" not in second["prompt_text"]
    assert "O2" not in second["prompt_text"]
    assert "A3" in decoded(second["response_tokens"])
    assert "O3_TERMINAL" not in decoded(second["response_tokens"])

    assert len(agent.trajectory.steps) == 4
    assert agent.trajectory.steps[2].model_response == responses[2]
    assert agent.trajectory.steps[2].next_observation == ACKNOWLEDGEMENT
    assert agent.trajectory.steps[3].next_observation.startswith("O3_TERMINAL")


def test_two_compressions_create_three_segments_with_distinct_real_prompts():
    responses = [
        compression_response("FIRST_RAW_THOUGHT", "SUMMARY_S1"),
        action_response("BETWEEN"),
        compression_response("SECOND_RAW_THOUGHT", "SUMMARY_S2"),
        action_response("AFTER_S2"),
    ]
    env = FakeEnv(
        [
            ("BETWEEN_OBSERVATION", 0.0, False, {}),
            ("FINAL_OBSERVATION", 0.0, True, {}),
        ]
    )
    agent = SWEAgent(enable_context_compression=True)

    result, _ = run_token_trajectory(agent, env, responses, max_steps=8, episode_id="episode-two-compressions")

    segments = result["policy_segments"]
    assert len(segments) == 3
    for index, segment in enumerate(segments):
        assert_segment_well_formed(segment, "episode-two-compressions")
        assert segment["segment_id"] == index

    assert "SUMMARY_S1" not in segments[0]["prompt_text"]
    assert "SUMMARY_S1" in segments[1]["prompt_text"]
    assert "SUMMARY_S2" not in segments[1]["prompt_text"]
    assert "SUMMARY_S2" in segments[2]["prompt_text"]
    assert "SUMMARY_S1" not in segments[2]["prompt_text"]
    assert ACKNOWLEDGEMENT in segments[1]["prompt_text"]
    assert ACKNOWLEDGEMENT in segments[2]["prompt_text"]
    assert "BETWEEN_OBSERVATION" not in segments[2]["prompt_text"]
    assert segments[0]["compression_index"] == 0
    assert segments[1]["compression_index"] == 1
    assert not segments[2]["ended_by_compression"]
    assert len(agent.trajectory.steps) == 4


def test_max_steps_finalizes_last_segment_without_pending_observation():
    responses = [action_response("A1"), action_response("A2")]
    env = FakeEnv(
        [
            ("O1_USED", 0.0, False, {}),
            ("O2_UNUSED_AT_MAX_STEPS", 0.0, False, {}),
        ]
    )
    agent = SWEAgent(enable_context_compression=True)

    result, _ = run_token_trajectory(agent, env, responses, max_steps=2)

    assert result["termination_reason"] == "MAX_STEPS"
    assert len(result["policy_segments"]) == 1
    response_text = decoded(result["policy_segments"][0]["response_tokens"])
    assert "O1_USED" in response_text
    assert "O2_UNUSED_AT_MAX_STEPS" not in response_text
    assert agent.trajectory.steps[-1].next_observation.startswith("O2_UNUSED_AT_MAX_STEPS")


def test_feature_disabled_keeps_baseline_result_shape_without_segmentation():
    response = action_response("BASELINE")
    env = FakeEnv([("BASELINE_DONE", 0.0, True, {})])
    agent = SWEAgent(enable_context_compression=False)

    result, _ = run_token_trajectory(agent, env, [response], max_steps=5)

    assert "episode_id" not in result
    assert "policy_segments" not in result
    assert set(("prompt_tokens", "response_tokens", "response_masks", "trajectory_reward")).issubset(result)
    assert env.step.call_count == 1
