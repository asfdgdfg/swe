import pytest

from rllm.agents.swe_agent import SWEAgent, parse_compress_context_action, parse_xml_response
from rllm.agents.system_prompts import (
    SWE_CONTEXT_COMPRESSION_FUNCTION_INVENTORY_ENTRY,
    OPENHANDS_SYSTEM_PROMPT,
    SWE_CONTEXT_COMPRESSION_TOOL_PROMPT,
    SWE_SYSTEM_PROMPT,
    SWEAGENT_SYSTEM_PROMPT,
)


@pytest.mark.parametrize(
    "summary",
    [
        "Task state is preserved.",
        """Task:
Fix the parser.

Remaining work:
Run the focused tests.""",
        "保留任务状态、已修改文件和剩余测试。",
        "Special characters: <tag> & $PATH 'quoted' \"double quoted\".",
        "Code:\n```python\ndef function_example():\n    return '<function text>'\n```",
        "  summary with surrounding whitespace  ",
    ],
)
def test_valid_compress_context_schema(summary):
    response = f"""Reasoning before the action.
<function=compress_context>
<parameter=summary>{summary}</parameter>
</function>"""

    thought, action = parse_xml_response(response)
    result = parse_compress_context_action(action)

    assert thought == "Reasoning before the action."
    assert action.function_name == "compress_context"
    assert result.is_compression is True
    assert result.is_valid is True
    assert result.summary == summary.strip()
    assert result.error is None


@pytest.mark.parametrize(
    ("parameters", "expected_error"),
    [
        ({}, "requires exactly one parameter"),
        ({"other": "value"}, "requires exactly one parameter"),
        ({"summary": "valid", "other": "value"}, "requires exactly one parameter"),
        ({"summary": ""}, "must be non-empty"),
        ({"summary": "  \n  "}, "must be non-empty"),
    ],
)
def test_invalid_compress_context_schema(parameters, expected_error):
    from r2egym.agenthub.action import Action as SWEAction

    result = parse_compress_context_action(SWEAction("compress_context", parameters))

    assert result.is_compression is True
    assert result.is_valid is False
    assert result.summary is None
    assert expected_error in result.error


def test_non_compression_action_is_not_reinterpreted():
    response = """Inspect the repository.
<function=execute_bash>
<parameter=command>rg -n 'function' .</parameter>
</function>"""

    thought, action = parse_xml_response(response)
    result = parse_compress_context_action(action)

    assert thought == "Inspect the repository."
    assert action.function_name == "execute_bash"
    assert action.parameters == {"command": "rg -n 'function' ."}
    assert result.is_compression is False
    assert result.is_valid is True
    assert result.summary is None
    assert result.error is None


def test_str_replace_editor_parser_regression():
    response = """Edit the file.
<function=str_replace_editor>
<parameter=command>str_replace</parameter>
<parameter=path>/testbed/example.py</parameter>
<parameter=old_str>old</parameter>
<parameter=new_str>new</parameter>
</function>"""

    thought, action = parse_xml_response(response)

    assert thought == "Edit the file."
    assert action.function_name == "str_replace_editor"
    assert action.parameters == {
        "command": "str_replace",
        "path": "/testbed/example.py",
        "old_str": "old",
        "new_str": "new",
    }


def test_malformed_xml_keeps_existing_parser_behavior():
    response = """Reasoning.
<function=compress_context>
<parameter=summary>unfinished"""

    thought, action = parse_xml_response(response)

    assert thought == response.strip()
    assert action.function_name == ""
    assert action.parameters == {}


@pytest.mark.parametrize(
    ("scaffold", "base_prompt"),
    [
        ("r2egym", SWE_SYSTEM_PROMPT),
        ("sweagent", SWEAGENT_SYSTEM_PROMPT),
        ("openhands", OPENHANDS_SYSTEM_PROMPT),
    ],
)
def test_feature_disabled_preserves_system_prompt_exactly(scaffold, base_prompt):
    assert SWEAgent(scaffold=scaffold).system_prompt == base_prompt
    assert SWEAgent(scaffold=scaffold, enable_context_compression=False).system_prompt == base_prompt


@pytest.mark.parametrize(
    ("scaffold", "base_prompt"),
    [
        ("r2egym", SWE_SYSTEM_PROMPT),
        ("sweagent", SWEAGENT_SYSTEM_PROMPT),
        ("openhands", OPENHANDS_SYSTEM_PROMPT),
    ],
)
def test_feature_enabled_appends_neutral_tool_prompt_after_scaffold_selection(scaffold, base_prompt):
    agent = SWEAgent(scaffold=scaffold, enable_context_compression=True)

    assert agent.system_prompt == f"{base_prompt}\n\n{SWE_CONTEXT_COMPRESSION_TOOL_PROMPT}"
    assert agent.system_prompt.count(SWE_CONTEXT_COMPRESSION_TOOL_PROMPT) == 1
    assert "compress_context" in agent.system_prompt


def test_inventory_prompt_exposes_compression_as_sweagent_function_4():
    agent = SWEAgent(
        scaffold="sweagent",
        enable_context_compression=True,
        context_compression_prompt_style="inventory",
    )
    prompt = agent.system_prompt
    assert prompt.count(SWE_CONTEXT_COMPRESSION_FUNCTION_INVENTORY_ENTRY) == 1
    assert SWE_CONTEXT_COMPRESSION_TOOL_PROMPT not in prompt
    assert prompt.index("---- END FUNCTION #3 ----") < prompt.index("---- BEGIN FUNCTION #4: compress_context ----")
    assert prompt.index("---- END FUNCTION #4 ----") < prompt.index("If you choose to call a function ONLY reply")


@pytest.mark.parametrize("scaffold", ["r2egym", "openhands"])
def test_inventory_prompt_style_is_isolated_to_sweagent(scaffold):
    with pytest.raises(ValueError, match="supports only scaffold='sweagent'"):
        SWEAgent(
            scaffold=scaffold,
            enable_context_compression=True,
            context_compression_prompt_style="inventory",
        )


def test_unknown_compression_prompt_style_fails_fast():
    with pytest.raises(ValueError, match="Invalid context_compression_prompt_style"):
        SWEAgent(
            scaffold="sweagent",
            enable_context_compression=True,
            context_compression_prompt_style="unknown",
        )


def test_feature_prompt_has_no_usage_signal_or_forcing_language():
    prompt = SWE_CONTEXT_COMPRESSION_TOOL_PROMPT.lower()

    assert "current context usage" not in prompt
    assert "token threshold" not in prompt
    assert "automatically" not in prompt
    assert "must call" not in prompt
