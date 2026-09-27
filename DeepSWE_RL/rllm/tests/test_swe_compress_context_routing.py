import asyncio
import concurrent.futures
from types import MethodType
from unittest.mock import MagicMock

import pytest

from rllm.agents.swe_agent import SWEAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine


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
    def __init__(self, step_result=("external observation", 0.0, False, {"source": "env"})):
        self.entry = {"instance_id": "routing-test"}
        self.idx = 0
        self.has_container = True
        self.step = MagicMock(return_value=step_result)
        self.close = MagicMock()

    def reset(self):
        return "Fix the test issue.", {}

    def compute_final_reward(self):
        return 0.0


def make_engine(agent, env, responses, max_steps=1):
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


def run_trajectory(agent, env, responses, max_steps=1):
    engine = make_engine(agent, env, responses, max_steps=max_steps)
    try:
        result = asyncio.run(engine.run_agent_trajectory_async(0, application_id="routing-test", mode="Text"))
    finally:
        engine.executor.shutdown(wait=True)
    return result


@pytest.mark.parametrize(
    "response",
    [
        """Inspect the repository.
<function=execute_bash>
<parameter=command>pwd</parameter>
</function>""",
        """Edit the file.
<function=str_replace_editor>
<parameter=command>view</parameter>
<parameter=path>/testbed/example.py</parameter>
</function>""",
    ],
)
def test_regular_actions_still_call_env_step(response):
    agent = SWEAgent(enable_context_compression=True)
    env = FakeEnv(step_result=("external observation", 0.0, True, {"source": "env"}))

    run_trajectory(agent, env, [response], max_steps=20)

    assert env.step.call_count == 1
    assert env.step.call_args.args[0] == agent.trajectory.steps[0].action
    assert agent.trajectory.steps[0].next_observation.startswith("external observation")
    assert agent.trajectory.steps[0].info["source"] == "env"


def test_valid_compression_bypasses_env_and_uses_normal_agent_update_path():
    response = """Preserve the current task state.
<function=compress_context>
<parameter=summary>Files inspected: example.py. Remaining work: implement the fix.</parameter>
</function>"""
    agent = SWEAgent(enable_context_compression=True)
    env = FakeEnv()

    run_trajectory(agent, env, [response])

    assert env.step.call_count == 0
    assert len(agent.trajectory.steps) == 1
    prior_step = agent.trajectory.steps[0]
    assert prior_step.thought == "Preserve the current task state."
    assert prior_step.action.startswith("<function=compress_context>")
    assert prior_step.model_response == response
    assert prior_step.reward == 0.0
    assert prior_step.done is False
    assert prior_step.next_observation == "Context compression completed. Continue the original task using the summary above."
    assert prior_step.info["max_steps"] == 1
    assert prior_step.info["cur_tokens"] == 0
    assert prior_step.info["internal_action"] == "compress_context"
    assert agent.messages[-1] == {"role": "tool", "content": "Context compression completed. Continue the original task using the summary above."}
    assert agent.cur_step.observation == agent.messages[-1]["content"]


@pytest.mark.parametrize(
    "action_xml",
    [
        """<function=compress_context>
</function>""",
        """<function=compress_context>
<parameter=summary>   </parameter>
</function>""",
        """<function=compress_context>
<parameter=summary>valid</parameter>
<parameter=extra>not allowed</parameter>
</function>""",
    ],
)
def test_invalid_compression_bypasses_env_and_continues(action_xml):
    agent = SWEAgent(enable_context_compression=True)
    env = FakeEnv()

    run_trajectory(agent, env, [action_xml])

    assert env.step.call_count == 0
    prior_step = agent.trajectory.steps[0]
    assert prior_step.reward == 0.0
    assert prior_step.done is False
    assert prior_step.next_observation.startswith("Invalid compress_context action:")
    assert agent.messages[-1]["role"] == "tool"
    assert agent.cur_step.observation.startswith("Invalid compress_context action:")


def test_disabled_feature_preserves_baseline_env_routing_for_compression_name():
    response = """Try an action unknown to the environment.
<function=compress_context>
<parameter=summary>State</parameter>
</function>"""
    agent = SWEAgent(enable_context_compression=False)
    env = FakeEnv(step_result=("environment handled it", 0.0, True, {"source": "env"}))

    run_trajectory(agent, env, [response], max_steps=20)

    assert env.step.call_count == 1
    assert agent.trajectory.steps[0].next_observation.startswith("environment handled it")
    assert agent.trajectory.steps[0].info["source"] == "env"


def test_last_parsed_action_is_overwritten_every_turn_and_reset():
    agent = SWEAgent(enable_context_compression=True)
    env = FakeEnv()
    compression = """Compress.
<function=compress_context>
<parameter=summary>State</parameter>
</function>"""
    execute = """Continue.
<function=execute_bash>
<parameter=command>pwd</parameter>
</function>"""

    run_trajectory(agent, env, [compression, execute], max_steps=2)

    assert env.step.call_count == 1
    assert len(agent.trajectory.steps) == 2
    assert agent.get_compression_action_parse_result().is_compression is False
    agent.reset()
    assert agent.get_compression_action_parse_result() is None


def test_agent_can_continue_with_another_model_turn_after_compression():
    agent = SWEAgent(enable_context_compression=True)
    env = FakeEnv()
    responses = [
        """Compress.
<function=compress_context>
<parameter=summary>State</parameter>
</function>""",
        """Continue normally.
<function=execute_bash>
<parameter=command>pwd</parameter>
</function>""",
    ]

    run_trajectory(agent, env, responses, max_steps=2)

    assert env.step.call_count == 1
    assert len(agent.trajectory.steps) == 2
    assert agent.trajectory.steps[0].action.startswith("<function=compress_context>")
    assert agent.trajectory.steps[1].action.startswith("<function=execute_bash>")
