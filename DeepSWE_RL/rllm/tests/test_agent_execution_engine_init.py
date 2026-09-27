from rllm.engine.agent_execution_engine import AgentExecutionEngine


class _Parser:
    pass


def _engine(**kwargs):
    return AgentExecutionEngine(
        engine_name="openai",
        chat_parser=_Parser(),
        rollout_engine_args={"api_key": "unit-test"},
        **kwargs,
    )


def test_explicit_env_response_retry_num_is_initialized():
    engine = _engine(env_response_retry_num=3)
    assert engine.env_response_retry_num == 3


def test_none_env_response_retry_num_uses_default():
    engine = _engine(env_response_retry_num=None)
    assert engine.env_response_retry_num == 3


def test_zero_env_response_retry_num_is_preserved():
    engine = _engine(env_response_retry_num=0)
    assert engine.env_response_retry_num == 0


def test_explicit_env_response_timeout_is_initialized():
    engine = _engine(env_response_timeout=3600)
    assert engine.env_response_timeout == 3600


def test_none_env_response_timeout_uses_default():
    engine = _engine(env_response_timeout=None)
    assert engine.env_response_timeout == 600
