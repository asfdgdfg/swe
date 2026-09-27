import asyncio
import concurrent.futures
from types import MethodType
from unittest.mock import MagicMock

import pytest

from rllm.agents.swe_agent import SWEAgent, parse_xml_response
from rllm.engine.agent_execution_engine import AgentExecutionEngine


ACKNOWLEDGEMENT = "Context compression completed. Continue the original task using the summary above."


class FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]


class FakeChatParser:
    assistant_token = "<assistant>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False):
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += self.assistant_token
        if len(messages) == 1 and messages[0]["role"] == "assistant":
            rendered = self.assistant_token + messages[0]["content"]
        return rendered


class FakeEnv:
    def __init__(self):
        self.entry = {"instance_id": "working-context-test"}
        self.idx = 0
        self.has_container = True
        self.step = MagicMock(return_value=("external observation", 0.0, True, {"source": "env"}))
        self.close = MagicMock()

    def reset(self):
        return "Fix the working-context test issue.", {}

    def compute_final_reward(self):
        return 0.0


def make_engine(agent, env, responses, max_steps):
    engine = object.__new__(AgentExecutionEngine)
    engine.agents = [agent]
    engine.envs = [env]
    engine.max_steps = max_steps
    engine.max_response_length = 100_000
    engine.max_prompt_length = 100_000
    engine.enforce_max_prompt_length = False
    engine.trajectory_timeout = 1_000
    engine.env_response_timeout = 10
    engine.env_response_retry_num = 2
    engine.overlong_filter = False
    engine.gamma = 0.2
    engine.tokenizer = FakeTokenizer()
    engine.chat_parser = FakeChatParser()
    engine.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    response_iter = iter(responses)

    async def fake_get_model_response(self, prompt, application_id, **kwargs):
        return next(response_iter)

    engine.get_model_response = MethodType(fake_get_model_response, engine)
    return engine


def run_step_trajectory(agent, env, responses, max_steps):
    engine = make_engine(agent, env, responses, max_steps)
    try:
        return asyncio.run(engine.run_agent_trajectory_async(0, application_id="working-context-test", mode="Step"))
    finally:
        engine.executor.shutdown(wait=True)


def initialize_agent(agent, max_steps=20):
    agent.reset()
    agent.update_from_env("Fix the original issue.", 0.0, False, {"max_steps": max_steps})


def add_action_and_observation(agent, thought, function_name, parameters, observation, max_steps=20):
    params = "\n".join(f"<parameter={name}>{value}</parameter>" for name, value in parameters.items())
    response = f"{thought}\n<function={function_name}>\n{params}\n</function>"
    agent.update_from_model(response)
    agent.update_from_env(observation, 0.0, False, {"max_steps": max_steps, "cur_tokens": 0})
    return response


def apply_valid_compression(agent, thought, summary, max_steps=20):
    response = f"{thought}\n<function=compress_context>\n<parameter=summary>{summary}</parameter>\n</function>"
    agent.update_from_model(response)
    result = agent.get_compression_action_parse_result()
    assert result.is_valid
    agent.apply_context_compression(result.summary)
    agent.update_from_env(
        ACKNOWLEDGEMENT,
        0.0,
        False,
        {"max_steps": max_steps, "cur_tokens": 0, "internal_action": "compress_context"},
    )
    return response


def test_one_compression_rebuilds_exact_working_context_and_preserves_trajectory():
    agent = SWEAgent(enable_context_compression=True)
    initialize_agent(agent)
    first_response = add_action_and_observation(agent, "First thought.", "execute_bash", {"command": "echo A1"}, "O1")
    second_response = add_action_and_observation(agent, "Second thought.", "execute_bash", {"command": "echo A2"}, "O2")
    compression_response = apply_valid_compression(agent, "Long compression thought that must be removed.", "S1: preserve the useful state.")

    assert [message["role"] for message in agent.messages] == ["system", "user", "assistant", "tool"]
    assert agent.messages[0]["content"] == agent.system_prompt
    assert agent.messages[1]["content"] == agent._original_task_message
    assert "This is step" not in agent.messages[1]["content"]
    assert agent.messages[2]["content"] == "<function=compress_context>\n  <parameter=summary>S1: preserve the useful state.</parameter>\n</function>"
    assert agent.messages[3] == {"role": "tool", "content": ACKNOWLEDGEMENT}
    working_context = "\n".join(message["content"] for message in agent.messages)
    assert "First thought" not in working_context
    assert "echo A1" not in working_context
    assert "Second thought" not in working_context
    assert "echo A2" not in working_context
    assert "O1" not in working_context
    assert "O2" not in working_context
    assert "Long compression thought" not in working_context

    assert len(agent.trajectory.steps) == 3
    assert agent.trajectory.steps[0].model_response == first_response
    assert agent.trajectory.steps[0].next_observation.startswith("O1")
    assert agent.trajectory.steps[1].model_response == second_response
    assert agent.trajectory.steps[1].next_observation.startswith("O2")
    assert agent.trajectory.steps[2].model_response == compression_response
    assert agent.trajectory.steps[2].next_observation == ACKNOWLEDGEMENT


@pytest.mark.parametrize(
    "summary",
    [
        "Line one.\nLine two.\nLine three.",
        "中文总结：保留文件、测试和剩余工作。",
        "```python\ndef function_example():\n    return '<tag>'\n```",
        "Run: rg -n 'function' . && python -m pytest",
        "XML-like text: <state><file>example.py</file></state>",
    ],
)
def test_summary_content_is_preserved_in_canonical_action(summary):
    agent = SWEAgent(enable_context_compression=True)
    initialize_agent(agent)
    apply_valid_compression(agent, "Discard this thought.", summary)

    _, canonical_action = parse_xml_response(agent.messages[2]["content"])
    assert canonical_action.function_name == "compress_context"
    assert canonical_action.parameters == {"summary": summary.strip()}


def test_compression_prompt_is_shorter_and_next_normal_action_uses_it():
    agent = SWEAgent(enable_context_compression=True)
    env = FakeEnv()
    long_observation = "large observation " * 200
    responses = [
        f"""Inspect.
<function=execute_bash>
<parameter=command>echo inspect</parameter>
</function>""",
        """Compress now.
<function=compress_context>
<parameter=summary>Only the essential state remains.</parameter>
</function>""",
        """Continue after compression.
<function=execute_bash>
<parameter=command>pwd</parameter>
</function>""",
    ]
    env.step.side_effect = [
        (long_observation, 0.0, False, {"source": "env"}),
        ("finished", 0.0, True, {"source": "env"}),
    ]

    result = run_step_trajectory(agent, env, responses, max_steps=20)

    assert env.step.call_count == 2
    prompt_before_compression = result["steps"][1]["prompt"]
    prompt_after_compression = result["steps"][2]["prompt"]
    tokenizer = FakeTokenizer()
    assert len(tokenizer.encode(prompt_after_compression)) < len(tokenizer.encode(prompt_before_compression))
    assert long_observation in prompt_before_compression
    assert long_observation not in prompt_after_compression
    assert "Only the essential state remains." in prompt_after_compression
    assert ACKNOWLEDGEMENT in prompt_after_compression


def test_second_compression_replaces_first_summary_but_preserves_both_steps():
    agent = SWEAgent(enable_context_compression=True)
    initialize_agent(agent)
    first_compression_response = apply_valid_compression(agent, "First compression.", "UNIQUE_SUMMARY_S1")
    add_action_and_observation(agent, "Continue after S1.", "execute_bash", {"command": "echo between"}, "between observation")
    second_compression_response = apply_valid_compression(agent, "Second compression.", "UNIQUE_SUMMARY_S2")

    working_context = "\n".join(message["content"] for message in agent.messages)
    assert "UNIQUE_SUMMARY_S2" in working_context
    assert "UNIQUE_SUMMARY_S1" not in working_context
    assert "between observation" not in working_context
    assert len(agent.trajectory.steps) == 3
    assert agent.trajectory.steps[0].model_response == first_compression_response
    assert agent.trajectory.steps[2].model_response == second_compression_response
    full_history = "\n".join(step.model_response for step in agent.trajectory.steps)
    assert "UNIQUE_SUMMARY_S1" in full_history
    assert "UNIQUE_SUMMARY_S2" in full_history


def test_disabled_feature_keeps_baseline_message_accumulation():
    agent = SWEAgent(enable_context_compression=False)
    initialize_agent(agent)
    response = add_action_and_observation(
        agent,
        "Compression-looking thought remains.",
        "compress_context",
        {"summary": "S1"},
        "environment handled the action",
    )

    assert agent._original_task_message is None
    assert len(agent.messages) == 4
    assert agent.messages[2] == {"role": "assistant", "content": response}
    assert agent.messages[3]["role"] == "tool"
    assert agent.messages[3]["content"].startswith("environment handled the action")


def test_invalid_compression_does_not_rebuild_working_context():
    agent = SWEAgent(enable_context_compression=True)
    initialize_agent(agent)
    add_action_and_observation(agent, "Keep this history.", "execute_bash", {"command": "echo keep"}, "keep observation")
    messages_before_invalid = list(agent.messages)
    invalid_response = "<function=compress_context>\n<parameter=summary>   </parameter>\n</function>"
    agent.update_from_model(invalid_response)
    result = agent.get_compression_action_parse_result()
    assert result.is_compression and not result.is_valid
    error_observation = f"Invalid compress_context action: {result.error}"
    agent.update_from_env(
        error_observation,
        0.0,
        False,
        {"max_steps": 20, "cur_tokens": 0, "internal_action": "compress_context"},
    )

    assert agent.messages[:-2] == messages_before_invalid
    assert agent.messages[-2] == {"role": "assistant", "content": invalid_response}
    assert agent.messages[-1] == {"role": "tool", "content": error_observation}
