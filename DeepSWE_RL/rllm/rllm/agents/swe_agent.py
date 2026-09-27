import json
import logging
import re
from dataclasses import dataclass

try:
    from r2egym.agenthub.action import Action as SWEAction
except ImportError:
    SWEAction = None

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.system_prompts import SWE_CONTEXT_COMPRESSION_FUNCTION_INVENTORY_ENTRY, SWE_CONTEXT_COMPRESSION_TOOL_PROMPT, SWE_SYSTEM_PROMPT, SWE_SYSTEM_PROMPT_FN_CALL, SWE_USER_PROMPT, SWE_USER_PROMPT_FN_CALL, SWEAGENT_SYSTEM_PROMPT, SWEAGENT_USER_PROMPT, OPENHANDS_SYSTEM_PROMPT, OPENHANDS_USER_PROMPT

TOKEN_WARNING_THRESHOLD = 128000
COMPRESS_CONTEXT_FUNCTION_NAME = "compress_context"
COMPRESS_CONTEXT_SUMMARY_PARAMETER = "summary"
CONTEXT_COMPRESSION_PROMPT_STYLES = {"additional", "inventory"}
CONTEXT_COMPRESSION_MODES = {"autonomous", "hybrid", "forced"}
SWEAGENT_FUNCTION_INVENTORY_END_ANCHOR = "---- END FUNCTION #3 ----"
SWEAGENT_FUNCTION_CALL_INSTRUCTIONS_ANCHOR = "If you choose to call a function ONLY reply in the following format with NO suffix:"


def add_context_compression_to_system_prompt(system_prompt: str, scaffold: str, prompt_style: str) -> str:
    """Expose compress_context without changing its parser or execution semantics."""
    if prompt_style not in CONTEXT_COMPRESSION_PROMPT_STYLES:
        raise ValueError(
            f"Invalid context_compression_prompt_style={prompt_style!r}; "
            f"expected one of {sorted(CONTEXT_COMPRESSION_PROMPT_STYLES)}"
        )
    if prompt_style == "additional":
        return f"{system_prompt}\n\n{SWE_CONTEXT_COMPRESSION_TOOL_PROMPT}"
    if scaffold != "sweagent":
        raise ValueError("context_compression_prompt_style='inventory' currently supports only scaffold='sweagent'")

    end_anchor = SWEAGENT_FUNCTION_INVENTORY_END_ANCHOR
    instructions_anchor = SWEAGENT_FUNCTION_CALL_INSTRUCTIONS_ANCHOR
    if system_prompt.count(end_anchor) != 1 or system_prompt.count(instructions_anchor) != 1:
        raise ValueError("SWEAgent function inventory anchors must each occur exactly once")
    end_pos = system_prompt.index(end_anchor) + len(end_anchor)
    instructions_pos = system_prompt.index(instructions_anchor)
    if end_pos >= instructions_pos:
        raise ValueError("SWEAgent function inventory anchors are out of order")
    return (
        f"{system_prompt[:end_pos]}\n\n\n"
        f"{SWE_CONTEXT_COMPRESSION_FUNCTION_INVENTORY_ENTRY}\n\n\n"
        f"{system_prompt[instructions_pos:]}"
    )


@dataclass(frozen=True)
class CompressionActionParseResult:
    is_compression: bool
    is_valid: bool
    summary: str | None = None
    error: str | None = None


def parse_compress_context_action(action: SWEAction) -> CompressionActionParseResult:
    """Validate the schema of an already-parsed compress_context action."""
    if action.function_name != COMPRESS_CONTEXT_FUNCTION_NAME:
        return CompressionActionParseResult(is_compression=False, is_valid=True)
    if set(action.parameters) != {COMPRESS_CONTEXT_SUMMARY_PARAMETER}:
        return CompressionActionParseResult(is_compression=True, is_valid=False, error="compress_context requires exactly one parameter named summary")
    summary = action.parameters[COMPRESS_CONTEXT_SUMMARY_PARAMETER]
    if not isinstance(summary, str) or not summary.strip():
        return CompressionActionParseResult(is_compression=True, is_valid=False, error="compress_context summary must be non-empty")
    return CompressionActionParseResult(is_compression=True, is_valid=True, summary=summary.strip())


def parse_oai_response(response):
    thought = response.choices[0].message.content
    if not thought:
        thought = ""
    try:
        function_name = response.choices[0].message.tool_calls[0].function.name
        parameters = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        action = SWEAction(function_name, parameters)
    except Exception:
        action = SWEAction(function_name="", parameters={})
    return thought, action


def parse_xml_response(response_text: str) -> tuple[str, SWEAction]:
    """
    Extracts:
    - thought: everything before the first <function=...> block
    - action: the entire first <function=...></function> block
    Returns (thought, action).
    """
    # Regex to match (non-greedily) from `<function=` up to the first `</function>`
    pattern = re.compile(r"(?s)(<function=.*?</function>)")
    match = pattern.search(response_text)

    if match:
        action = match.group(1)  # The entire <function=...></function> block
        thought = response_text[: match.start()]  # Everything before the block
    else:
        # If no match, treat entire text as "thought"
        thought = response_text
        action = ""

    # Strip leading/trailing whitespace
    thought = thought.strip()
    action = action.strip()

    # convert action to Action object
    action = SWEAction.from_string(action)

    return thought, action


logger = logging.getLogger(__name__)


class SWEAgent(BaseAgent):
    def __init__(self, use_fn_calling: bool = False, format_model_response: bool = False, scaffold: str = "r2egym", enable_context_compression: bool = False, context_compression_prompt_style: str = "additional", context_compression_mode: str = "autonomous", compression_warning_tokens: int = 32768, compression_force_tokens: int = 40960, compression_summary_max_tokens: int = 1024, compression_retry_num: int = 2, compression_force_min_response_budget: int = 2048):
        self.use_fn_calling = use_fn_calling
        self.format_model_response = format_model_response
        self.scaffold = scaffold
        self.enable_context_compression = enable_context_compression
        self.context_compression_prompt_style = context_compression_prompt_style
        if context_compression_mode not in CONTEXT_COMPRESSION_MODES:
            raise ValueError(f"Invalid context_compression_mode={context_compression_mode!r}; expected one of {sorted(CONTEXT_COMPRESSION_MODES)}")
        if compression_warning_tokens < 0 or compression_force_tokens <= 0:
            raise ValueError("Compression token thresholds must be non-negative and force must be positive")
        if compression_warning_tokens >= compression_force_tokens:
            raise ValueError("compression_warning_tokens must be smaller than compression_force_tokens")
        if compression_summary_max_tokens <= 0 or compression_retry_num < 0:
            raise ValueError("compression_summary_max_tokens must be positive and compression_retry_num non-negative")
        if compression_force_min_response_budget <= 0:
            raise ValueError("compression_force_min_response_budget must be positive")
        self.context_compression_mode = context_compression_mode
        self.compression_warning_tokens = compression_warning_tokens
        self.compression_force_tokens = compression_force_tokens
        self.compression_summary_max_tokens = compression_summary_max_tokens
        self.compression_retry_num = compression_retry_num
        self.compression_force_min_response_budget = compression_force_min_response_budget

        print(f"!!!!!!! sweagent use fn_calling: {use_fn_calling}")

        print(f"!!!!!!!!!!!!!!!!!!!!!!!!! scaffold: {scaffold}")
        assert scaffold in ["r2egym", "sweagent", "openhands"], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"
        self.system_prompt = SWE_SYSTEM_PROMPT_FN_CALL if use_fn_calling else SWE_SYSTEM_PROMPT
        if scaffold == "sweagent":
            self.system_prompt = SWEAGENT_SYSTEM_PROMPT
        if scaffold == "openhands":
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!! scaffold: {scaffold}, using OPENHANDS_SYSTEM_PROMPT")
            self.system_prompt = OPENHANDS_SYSTEM_PROMPT
            # print(f"!!!!!!!!!!!!!!!!!!!!!!!!! OPENHANDS_SYSTEM_PROMPT: \n {self.system_prompt}")

        if enable_context_compression:
            self.system_prompt = add_context_compression_to_system_prompt(
                self.system_prompt,
                scaffold=scaffold,
                prompt_style=context_compression_prompt_style,
            )

        self.user_prompt_template = SWE_USER_PROMPT_FN_CALL if use_fn_calling else SWE_USER_PROMPT
        if scaffold == "sweagent":
            self.user_prompt_template = SWEAGENT_USER_PROMPT
        if scaffold == "openhands":
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!! scaffold: {scaffold}, using OPENHANDS_USER_PROMPT")
            self.user_prompt_template = OPENHANDS_USER_PROMPT
            # print(f"!!!!!!!!!!!!!!!!!!!!!!!!! OPENHANDS_USER_PROMPT: \n {self.system_prompt}")

        self._trajectory = Trajectory()
        self.reset()

    def process_model_response(self, response: str) -> tuple[str, str]:
        """
        Processes the model's response to extract thought and action components.

        Parses the response using either function calling or XML parsing based on agent configuration.

        Args:
            response (str): The raw text response from the model.

        Returns:
            Tuple[str, str]: A tuple containing:
                - The action string in XML format
                - The processed response (may be reformatted if self.format_model_response is True)
        """
        if self.use_fn_calling:
            thought, action = parse_oai_response(response)
        else:
            thought, action = parse_xml_response(response)

        action_str = action.to_xml_string()
        if self.format_model_response:
            response = f"{thought}\n\n{action_str}"
        return action.to_xml_string(), {
            "thought": thought,
        }

    def update_from_env(self, observation, reward, done, info):
        # If the first step in environment, we need to update the state from the environment
        print(f"now in update_from_env")

        # rollout init
        user_prompt_flag = False
        if self._trajectory.steps:
            print(f"now in update_from_env self._trajectory.step")
            observation = str(observation)
            print(f"now in update_from_env self._trajectory.step done")
        else:
            print(f"now in update_from_env self._trajectory.step else")
            # print(f"now in update_from_env self._trajectory.step else: observation: {observation}")
            observation = str(observation)
            # print(f"now in update_from_env self._trajectory.step else: observation: {observation}")
            # print(f"now in update_from_env self._trajectory.step else: self.user_prompt_template: {self.user_prompt_template}")
            observation = (
                self.user_prompt_template
                .replace('[problem_statement]', observation)
                .replace('{problem_statement}', observation)
                .replace('[working_dir]', '/testbed')
                .replace('{working_dir}', '/testbed')
            )
            if '[problem_statement]' in observation or '{problem_statement}' in observation:
                raise ValueError('problem_statement placeholder was not substituted')
            user_prompt_flag = True
            print(f"SWE placeholder substitution: problem_statement={'[problem_statement]' not in observation}, working_dir={'[working_dir]' not in observation}")
            print(f"now in update_from_env self._trajectory.step else done")
            if self.enable_context_compression:
                self._original_task_message = observation


        print(f"now in update_from_env has get obs")
        max_steps = info.get("max_steps", None)
        is_internal_action = info.get("internal_action") is not None
        if max_steps and not is_internal_action:
            remaining_steps = max_steps - self.step - 1
            step_count = self.step+1
            if remaining_steps > 0:
                stepcount_message = (
                                f"This is step {step_count} of a maximum of {max_steps}. "
                                f"Steps Remaining: {remaining_steps}."
                            )
                # observation += f"\nSteps Remaining: {remaining_steps}"
                observation += f"\n{stepcount_message}"

                print(f"!!! now in update_from_env has get obs max_steps")
            else:
                observation += "\nYou have reached the maximum number of steps. Please submit your answer NOW."

        print(f"now in update_from_env has get obs done")
        cur_tokens = info.get("cur_tokens", None)
        if cur_tokens is not None and cur_tokens >= TOKEN_WARNING_THRESHOLD:
            observation += "\nYou are running out of tokens. Please submit your answer NOW."

        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info
        print(f"!!!!!!!!!!!!! now in update_from_env to step env")

        # self.messages.append({"role": "user", "content": observation})
        if user_prompt_flag:
            self.messages.append({"role": "user", "content": observation})
        else:
            self.messages.append({"role": "tool", "content": observation})

        print(f"!!!!!!!!!!!!!! now in update_from_env to step env obs")
        self.cur_step = Step(observation=observation)
        print(f"!!!!!!!!!!!!!! now in update_from_env to step env obs done")

    def update_from_model(self, response: str, **kwargs):
        """
        Updates the agent's internal state after an environment step.

        This function is called during environment interaction to incorporate the latest action's
        outcome into the agent's learning process.

        Args:
            response (str): The response from the model.

        Returns:
            None
        """

        print(f"!!!!!!!!!!!!!! now in update_from_model")
        self._trajectory.steps.append(self.cur_step)
        parsed_response = kwargs.get("parsed_response")
        if parsed_response is not None:
            thought, action = parsed_response
        elif self.use_fn_calling:
            print("!!!!!!!!!!!!!! now in update_from_model fn_calling")
            thought, action = parse_oai_response(response)
            print("!!!!!!!!!!!!!! now in update_from_model fn_calling done")
        else:
            print("!!!!!!!!!!!!!! now in update_from_model xml")
            thought, action = parse_xml_response(response)
            print("!!!!!!!!!!!!!! now in update_from_model xml done")
        self._last_parsed_action = action
        action_str = action.to_xml_string()
        assert self._trajectory.steps, "Trajectory should not be empty when update_from_model is called."

        # Update Trajectory
        cur_step = self._trajectory.steps[-1]
        cur_step.thought = thought
        cur_step.action = action_str
        cur_step.model_response = response

        # Update Chat Completions
        if self.format_model_response:
            self.messages.append({"role": "assistant", "content": f"{thought}\n\n{action_str}"})
        else:
            self.messages.append({"role": "assistant", "content": response})
        self.step += 1
        return Action(action=cur_step.action)

    def parse_model_response(self, response: str):
        """Parse a response without mutating messages or the formal trajectory."""
        if self.use_fn_calling:
            return parse_oai_response(response)
        return parse_xml_response(response)

    def apply_context_compression(self, summary: str) -> None:
        """Replace only the model's working context with the supplied summary."""
        if not self.enable_context_compression:
            raise RuntimeError("Context compression is disabled")
        if self._original_task_message is None:
            raise RuntimeError("Original task message is unavailable")

        canonical_action = SWEAction(
            function_name=COMPRESS_CONTEXT_FUNCTION_NAME,
            parameters={COMPRESS_CONTEXT_SUMMARY_PARAMETER: summary},
        ).to_xml_string()
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._original_task_message},
            {"role": "assistant", "content": canonical_action},
        ]

    def get_compression_action_parse_result(self) -> CompressionActionParseResult | None:
        """Return compression schema validation for the most recently parsed action."""
        if self._last_parsed_action is None:
            return None
        return parse_compress_context_action(self._last_parsed_action)

    def validate_compression_action(self, action: SWEAction) -> CompressionActionParseResult:
        """Validate a parsed action without committing it to trajectory state."""
        return parse_compress_context_action(action)

    def get_current_state(self) -> Step:
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]

    def reset(self):
        self._trajectory = Trajectory()
        self._last_parsed_action = None
        self._original_task_message = None
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        self.step = 0

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self):
        return self.messages
