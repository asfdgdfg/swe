import asyncio
import concurrent.futures
import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np
import openai
import torch
from openai.types import Completion

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.utils import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.env_utils import (
    compute_mc_return,
    compute_trajectory_reward,
)
from rllm.misc import colorful_print
from rllm.parser.chat_template.parser import ChatTemplateParser
from rllm.router.router import Router
from rllm.agents.system_prompts import (
    SWE_CONTEXT_COMPRESSION_REQUIRED_PROMPT,
    SWE_CONTEXT_COMPRESSION_WARNING_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class PolicySegment:
    episode_id: str
    segment_id: int
    prompt_text: str
    prompt_tokens: list[int]
    response_tokens: list[int] = field(default_factory=list)
    policy_mask: list[int] = field(default_factory=list)
    start_model_call: int = 0
    end_model_call: int | None = None
    ended_by_compression: bool = False
    compression_index: int | None = None
    compression_source: str | None = None


class AgentExecutionEngine:
    def __init__(
        self,
        engine_name="openai",
        tokenizer=None,
        rollout_engine=None,
        chat_parser=None,
        n_parallel_agents=1,
        trajectory_timeout=None,
        env_response_timeout=None,
        env_response_retry_num = None,
        gamma=0.2,
        api_retries=3,
        retry_limit=3,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        config=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=64,
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if rollout_engine_args is None:
            rollout_engine_args = {}
        if env_args is None:
            env_args = {}

        self.config = config
        self.rollout_engine = rollout_engine
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.overlong_filter = overlong_filter

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.api_retries = api_retries
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args

        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]

        self.trajectory_timeout = trajectory_timeout
        if not trajectory_timeout:
            self.trajectory_timeout = int(1e9)
        self.env_response_timeout = (
            int(600) if env_response_timeout is None else env_response_timeout
        )
        self.env_response_retry_num = (
            int(3) if env_response_retry_num is None else env_response_retry_num
        )
        if env_class is not None:
            assert env_class.is_multithread_safe(), "Environment must be multithread safe for async engine"
        # rollout engine args
        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", {})

        assert self.engine_name in ["openai", "verl"], "Currently only openai and verl are supported as rollout engine"
        if self.engine_name == "openai":
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(**self.rollout_engine_args)
            # Disable httpx INFO logs that show HTTP requests
            logging.getLogger("httpx").setLevel(logging.WARNING)
        elif self.engine_name == "verl":
            # All generation is done via scheduler. Currently only works for verl
            self.server_addresses = getattr(self.rollout_engine, "server_addresses", [])
            self.router = Router(config=self.config, tokenizer=self.tokenizer, addresses=self.server_addresses)

        # Create a thread pool executor for environment interactions (i.e. step, reset, close)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
        else:
            self.chat_parser = chat_parser

    async def get_model_response(self, prompt, application_id, **kwargs):
        """
        Compute model response asynchronously based on the engine type.

        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.

        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            **kwargs: Additional arguments to pass to the model

        Returns:
            The model's response text

        Raises:
            NotImplementedError: If the engine type is not supported
        """
        if self.engine_name == "openai":
            return await self._get_openai_async(prompt, application_id, **kwargs)
        elif self.engine_name == "verl":
            return await self._get_verl_async(prompt, application_id, **kwargs)
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def update_envs_and_agents(self, envs, agents):
        """
        Update the environments and agents.

        Args:
            envs: List of environments to use
            agents: List of agents to use
        """
        assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
        self.envs = envs
        # For keeping track of the environment index in the batch.
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents
        self.n_parallel_agents = len(envs)
        logger.info(f"in update_envs_and_agents len(envs):{len(envs)}, sets n_parallel_agents to {self.n_parallel_agents}")

    async def _get_verl_async(self, prompt, application_id, **kwargs):
        batch = self._convert_prompt_verl([prompt], **kwargs)

        if "max_tokens" in kwargs:
            batch.meta_info["max_tokens"] = kwargs["max_tokens"]

        output = await self.router.generate_sequences(batch, application_id=application_id, **kwargs)

        attn = output.batch["attention_mask"][0, self.max_prompt_length :]
        tokens = output.batch["responses"][0]

        # Find last index where attention == 1
        non_pad_indices = (attn == 1).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) == 0:
            trimmed = tokens[:0]  # empty
        else:
            last_valid_idx = non_pad_indices[-1].item()
            trimmed = tokens[: last_valid_idx + 1]  # include the last valid token

        response = self.tokenizer.decode(trimmed, skip_special_tokens=False)

        pad_token = self.tokenizer.pad_token
        eos_token = self.tokenizer.eos_token
        response = response.replace(pad_token, "").replace(eos_token, "")
        return response

    async def _get_openai_async(self, prompt, _, **kwargs):
        """
        Get action from OpenAI API asynchronously with retry logic.

        Args:
            prompt: The input prompt in text format for completions API
            application_id: Unique identifier for the application (unused for OpenAI)
            **kwargs: Additional arguments to pass to the OpenAI API

        Returns:
            The response from OpenAI API
        """

        async def get_response(prompt_text: str):
            retries = self.api_retries
            while retries > 0:
                try:
                    response = await self.client.completions.create(
                        prompt=prompt_text,
                        timeout=3600,
                        **self.sampling_params,
                        **kwargs,
                    )
                    return response
                except openai.RateLimitError:
                    retries -= 1
                    if retries == 0:
                        return "Error: Rate limit reached and retries exhausted."
                    logger.info("Sleep for 5 seconds for API limit.")
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error("Error: %s", e)
                    return f"Error processing content: {e}"

        # If prompt is in chat format, convert it to text format
        prompt_text = prompt
        if isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):
            prompt_text = self.chat_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True)

        response = await get_response(prompt_text)
        if isinstance(response, Completion):
            response = response.choices[0].text
        return response

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run a single agent's trajectory asynchronously"""
        print(f"!!!!!!!!! now in run_agent_trajectory_async")
        agent = self.agents[idx]
        env = self.envs[idx]
        # env_id = env.env_id

        termination_reason = None
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        env_tokens_len = 0
        model_gen_tokens_len = 0
        total_time = 0.0
        reward_time = 0.0
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        # Policy segmentation is an additive, compression-only data path. The
        # baseline flattened trajectory below remains unchanged.
        segmentation_enabled = bool(getattr(agent, "enable_context_compression", False))
        policy_segments = [] if segmentation_enabled else None
        active_policy_segment = None
        pending_segment_observation_tokens = []
        compression_index = 0
        autonomous_compression_calls = 0
        forced_compression_calls = 0
        autonomous_first_compression_position = None
        forced_first_compression_position = None
        forced_compression_retries = []
        forced_compression_failed = 0
        forced_compression_skipped_insufficient_budget = 0
        forced_compression_candidates = []
        compression_events = []

        # Generation backends budget raw model tokens, while the training
        # representation serializes every assistant message through the chat
        # parser (for Qwen this also appends ``<|im_end|>\n``).  Derive that
        # deterministic serialization overhead from the active parser and
        # tokenizer rather than assuming a model-specific constant.
        empty_assistant_tokens, _ = convert_messages_to_tokens_and_masks(
            [{"role": "assistant", "content": ""}],
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=False,
            contains_generation_msg=False,
        )
        assistant_serialization_overhead = len(empty_assistant_tokens)

        def start_policy_segment(prompt_messages, model_call_idx):
            prompt_text = self.chat_parser.parse(
                prompt_messages,
                add_generation_prompt=True,
                is_first_msg=True,
            )
            segment_prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            return PolicySegment(
                episode_id=application_id,
                segment_id=len(policy_segments),
                prompt_text=prompt_text,
                prompt_tokens=list(segment_prompt_tokens),
                start_model_call=model_call_idx,
            )

        def finalize_policy_segment(ended_by_compression=False, segment_compression_index=None, compression_source=None):
            nonlocal active_policy_segment
            if active_policy_segment is None:
                return
            active_policy_segment.ended_by_compression = ended_by_compression
            active_policy_segment.compression_index = segment_compression_index
            active_policy_segment.compression_source = compression_source
            assert len(active_policy_segment.response_tokens) == len(active_policy_segment.policy_mask)
            policy_segments.append(active_policy_segment)
            active_policy_segment = None

        # for step return
        episode_steps = []

        # Reset environment with the task using the executor
        loop = asyncio.get_event_loop()
        print(f"!!!!!! now resetting env")
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        print(f"!!!!!! now resetting env done")
        info["max_steps"] = self.max_steps

        if not env.has_container:
            termination_reason = "NO_CONTAINER"
        
        # Reset agent
        agent.reset()
        # Update agent internal state from environment.
        print(f"!!!!!! now updating agent from env")
        agent.update_from_env(
            observation=observation,  # Raw observation from environment
            reward=0.0,
            done=False,
            info=info
        )
        print(f"!!!!!! now updating agent from env done")
        messages = agent.chat_completions
        print(f"!!!!!! now updating agent from chat_completions")
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=True, contains_generation_msg=True)
        # print(f"2222222222")
        prompt_token_len = len(prompt_tokens)
        # print(f"3333333333")
        # Note, this should never happen!
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            termination_reason = "PROMPT_TRUNCATION"
            logger.info(f"Trajectory {idx}: initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length}")       
        
        process_terminated = False
        
        for step_idx in range(self.max_steps):
            if termination_reason == "NO_CONTAINER":
                chat_completions_messages = agent.chat_completions
                assistant_message = {'role': 'assistant', 'content': 'env initialization failed, I will stop this message!'}
                chat_completions_messages.append(assistant_message)
                assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks([assistant_message], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=False)
                response_tokens.extend(assistant_msg_tokens) 
                response_masks.extend(assistant_msg_masks)
                process_terminated = True
                info_of_this_traj = ""
                if "instance_id" in env.entry:
                    info_of_this_traj += f"instance_id:{env.entry['instance_id']}"
                if "docker_image" in env.entry:
                    info_of_this_traj += f"docker_image:{env.entry['docker_image']}"

                logger.info(f"Trajectory {idx}, info is {info_of_this_traj}: initial container failed, has exit!!!")
                break

            if termination_reason == "PROMPT_TRUNCATION":
                chat_completions_messages = agent.chat_completions
                assistant_message = {'role': 'assistant', 'content': 'prompt is too long, I will stop this message!'}
                chat_completions_messages.append(assistant_message)
                assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks([assistant_message], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=False)
                response_tokens.extend(assistant_msg_tokens) 
                response_masks.extend(assistant_msg_masks)
                process_terminated = True
                info_of_this_traj = ""
                if "instance_id" in env.entry:
                    info_of_this_traj += f"instance_id:{env.entry['instance_id']}"
                if "docker_image" in env.entry:
                    info_of_this_traj += f"docker_image:{env.entry['docker_image']}"

                logger.info(f"Trajectory {idx}, info is {info_of_this_traj}: initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length},has exit!!!")
                break

            # Get action from agent
            print(f"now in agent.get_action step: {step_idx}")
            prompt_messages = agent.chat_completions.copy()
            base_prompt_text = self.chat_parser.parse(
                prompt_messages, add_generation_prompt=True, is_first_msg=True
            )
            current_working_prompt_tokens = len(
                self.tokenizer.encode(base_prompt_text, add_special_tokens=False)
            )
            compression_mode = getattr(agent, "context_compression_mode", "autonomous")
            forced_compression = bool(
                segmentation_enabled
                and compression_mode in {"forced", "hybrid"}
                and current_working_prompt_tokens >= getattr(agent, "compression_force_tokens", 40960)
            )
            remaining_response_budget = self.max_response_length - response_token_len
            force_min_response_budget = getattr(
                agent, "compression_force_min_response_budget", 2048
            )
            force_candidate = None
            if forced_compression:
                force_candidate = {
                    "source": "forced_threshold_candidate",
                    "task_id": env.entry.get("instance_id"),
                    "episode_id": application_id,
                    "step_idx": step_idx,
                    "working_prompt_tokens": current_working_prompt_tokens,
                    "response_token_len": response_token_len,
                    "remaining_response_budget": remaining_response_budget,
                    "force_min_response_budget": force_min_response_budget,
                    "summary_tokens": None,
                    "working_prompt_after": None,
                    "status": "pending",
                }
                forced_compression_candidates.append(force_candidate)
                logger.info(
                    "Forced compression candidate: "
                    f"step_idx={step_idx}, "
                    f"working_prompt_tokens={current_working_prompt_tokens}, "
                    f"response_token_len={response_token_len}, "
                    f"remaining_response_budget={remaining_response_budget}, "
                    f"force_min_response_budget={force_min_response_budget}"
                )
                if remaining_response_budget < force_min_response_budget:
                    force_candidate["status"] = "skipped_insufficient_budget"
                    forced_compression_skipped_insufficient_budget += 1
                    termination_reason = "TRUNCATION"
                    reward = 0.0
                    if agent.trajectory.steps:
                        agent.get_current_state().done = True
                    break
            compression_warning = bool(
                segmentation_enabled
                and compression_mode == "hybrid"
                and not forced_compression
                and current_working_prompt_tokens >= getattr(agent, "compression_warning_tokens", 32768)
            )
            if forced_compression:
                prompt_messages = prompt_messages + [
                    {"role": "system", "content": SWE_CONTEXT_COMPRESSION_REQUIRED_PROMPT}
                ]
            elif compression_warning:
                prompt_messages = prompt_messages + [
                    {"role": "system", "content": SWE_CONTEXT_COMPRESSION_WARNING_PROMPT}
                ]
            # Max remaining tokens left for the response
            # For enforced max prompt at each step, no need to deduct here
            if not self.enforce_max_prompt_length:
                remaining_training_budget = self.max_response_length - response_token_len
                if remaining_training_budget <= assistant_serialization_overhead:
                    # There is not enough room for a complete serialized
                    # assistant message.  Do not generate a response that
                    # would necessarily exceed the training representation.
                    termination_reason = "TRUNCATION"
                    if agent.trajectory.steps:
                        cur_step = agent.get_current_state()
                        cur_step.done = True
                    break
                max_tokens = remaining_training_budget - assistant_serialization_overhead
            else:
                max_tokens = self.max_response_length

                # since max prompt is enforced, we filter out too long prompts.
                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

            kwargs["max_tokens"] = max_tokens

            response = None
            forced_parsed_response = None
            forced_budget_exhausted = False
            max_generation_attempts = (
                getattr(agent, "compression_retry_num", 2) + 1 if forced_compression else 1
            )
            for forced_attempt in range(max_generation_attempts):
                if forced_compression:
                    retry_remaining_response_budget = (
                        self.max_response_length - response_token_len
                    )
                    if retry_remaining_response_budget < force_min_response_budget:
                        if force_candidate is not None:
                            force_candidate["status"] = "skipped_insufficient_budget"
                        forced_compression_skipped_insufficient_budget += 1
                        termination_reason = "TRUNCATION"
                        reward = 0.0
                        forced_budget_exhausted = True
                        if agent.trajectory.steps:
                            agent.get_current_state().done = True
                        break
                start_time = time.time()
                print(f"!!!!!! now getting model response for step: {step_idx}")
                candidate_response = await self.get_model_response(
                    prompt_messages, application_id, **kwargs
                )
                delta_time = time.time() - start_time
                llm_time += delta_time
                total_time += delta_time
                if not forced_compression:
                    response = candidate_response
                    break

                retry_error = None
                try:
                    candidate_parsed = agent.parse_model_response(candidate_response)
                    candidate_compression = agent.validate_compression_action(candidate_parsed[1])
                    if not candidate_compression.is_compression or not candidate_compression.is_valid:
                        retry_error = candidate_compression.error or "response did not call compress_context"
                    else:
                        summary_tokens = self.tokenizer.encode(
                            candidate_compression.summary, add_special_tokens=False
                        )
                        if len(summary_tokens) > getattr(agent, "compression_summary_max_tokens", 1024):
                            retry_error = (
                                f"summary has {len(summary_tokens)} tokens, exceeding "
                                f"compression_summary_max_tokens={agent.compression_summary_max_tokens}"
                            )
                except Exception as exc:
                    retry_error = f"compression response parse failed: {exc}"

                if retry_error is None:
                    response = candidate_response
                    forced_parsed_response = candidate_parsed
                    break
                forced_compression_retries.append(
                    {
                        "step_idx": step_idx,
                        "attempt": forced_attempt + 1,
                        "error": retry_error,
                        "response": candidate_response,
                    }
                )

            if forced_budget_exhausted:
                break

            if response is None:
                forced_compression_failed += 1
                if force_candidate is not None:
                    force_candidate["status"] = "failed_retries_exhausted"
                termination_reason = "FORCED_COMPRESSION_FAILED"
                done = True
                reward = 0.0
                if agent.trajectory.steps:
                    agent.get_current_state().done = True
                break

            if segmentation_enabled:
                # Commit segment state only after a response has passed forced
                # compression validation. Failed retries are debug-only and do
                # not create a prompt row or consume a pending observation.
                if active_policy_segment is None:
                    active_policy_segment = start_policy_segment(prompt_messages, step_idx)
                elif pending_segment_observation_tokens:
                    active_policy_segment.response_tokens.extend(pending_segment_observation_tokens)
                    active_policy_segment.policy_mask.extend([0] * len(pending_segment_observation_tokens))
                    pending_segment_observation_tokens = []
            # Update steps
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response,
            }
            episode_steps.append(prompt_response_pair)

            # This is the single canonical assistant representation for the
            # current model call.  Both the flattened rollout and the optional
            # PolicySegment path reuse these exact token IDs and masks.
            raw_assistant_message = {"role": "assistant", "content": response}
            canonical_assistant_tokens, canonical_assistant_masks = (
                convert_messages_to_tokens_and_masks(
                    [raw_assistant_message],
                    tokenizer=self.tokenizer,
                    parser=self.chat_parser,
                    contains_first_msg=False,
                    contains_generation_msg=False,
                )
            )
            if len(canonical_assistant_tokens) > self.max_response_length - response_token_len:
                raise ValueError(
                    "Serialized assistant response exceeds the remaining training budget: "
                    f"serialized_length={len(canonical_assistant_tokens)}, "
                    f"remaining_training_budget={self.max_response_length - response_token_len}, "
                    f"serialization_overhead={assistant_serialization_overhead}"
                )

            # Update agent with model response
            action: Action = agent.update_from_model(
                response, parsed_response=forced_parsed_response
            )
            action = action.action

            compression_result = None
            if getattr(agent, "enable_context_compression", False):
                get_compression_result = getattr(agent, "get_compression_action_parse_result", None)
                if get_compression_result is not None:
                    compression_result = get_compression_result()
            is_internal_compression = compression_result is not None and compression_result.is_compression
            compression_source = None
            current_compression_event = None
            if is_internal_compression and compression_result.is_valid:
                compression_source = "forced_threshold" if forced_compression else "autonomous"
                summary_token_count = len(
                    self.tokenizer.encode(
                        compression_result.summary, add_special_tokens=False
                    )
                )
                if force_candidate is not None:
                    current_compression_event = force_candidate
                    current_compression_event.update(
                        {
                            "source": compression_source,
                            "summary_tokens": summary_token_count,
                            "status": "completed",
                        }
                    )
                else:
                    current_compression_event = {
                        "source": compression_source,
                        "task_id": env.entry.get("instance_id"),
                        "episode_id": application_id,
                        "step_idx": step_idx,
                        "working_prompt_tokens": current_working_prompt_tokens,
                        "response_token_len": response_token_len,
                        "remaining_response_budget": self.max_response_length - response_token_len,
                        "force_min_response_budget": force_min_response_budget,
                        "summary_tokens": summary_token_count,
                        "working_prompt_after": None,
                        "status": "completed",
                    }
                compression_events.append(current_compression_event)
                if forced_compression:
                    forced_compression_calls += 1
                    if forced_first_compression_position is None:
                        forced_first_compression_position = response_token_len
                else:
                    autonomous_compression_calls += 1
                    if autonomous_first_compression_position is None:
                        autonomous_first_compression_position = response_token_len

            if segmentation_enabled:
                # Always encode the raw model response. In the compression case,
                # agent.messages has already been canonicalized later and cannot
                # be used without losing the model's original thought.
                active_policy_segment.response_tokens.extend(canonical_assistant_tokens)
                active_policy_segment.policy_mask.extend(canonical_assistant_masks)
                active_policy_segment.end_model_call = step_idx

                if is_internal_compression and compression_result.is_valid:
                    # The compression response is the last policy output under
                    # the pre-compression context. The acknowledgement belongs
                    # only to the next real model call's prompt.
                    finalize_policy_segment(
                        ended_by_compression=True,
                        segment_compression_index=compression_index,
                        compression_source=compression_source,
                    )
                    compression_index += 1

            # print(f"now in step: action: {action}")

            # Take step in environment using the executor
            start_time = time.time()
            flag_get_env_response = False
            if is_internal_compression:
                if compression_result.is_valid:
                    agent.apply_context_compression(compression_result.summary)
                    next_observation = "Context compression completed. Continue the original task using the summary above."
                else:
                    next_observation = f"Invalid compress_context action: {compression_result.error}"
                reward = 0.0
                done = False
                info = {
                    "internal_action": "compress_context",
                    "compression_source": compression_source,
                    "working_prompt_tokens_before_compression": current_working_prompt_tokens,
                }
                flag_get_env_response = True

            env_response_attempts = 0 if is_internal_compression else self.env_response_retry_num
            for retry_num_get_env_response in range(env_response_attempts):
                try:
                    print(f"!!!!!!! now in env.step step: {step_idx}")
                    # next_observation, reward, done, info = await asyncio.wait_for(loop.run_in_executor(self.executor, env.step, action), timeout=(self.trajectory_timeout - total_time))
                    step_result = await asyncio.wait_for(
                        loop.run_in_executor(self.executor, env.step, action),
                        # timeout=(self.trajectory_timeout - total_time)
                        timeout=(self.env_response_timeout)
                    )
                    # print("now in step_result", step_result)

                    if isinstance(step_result, tuple) and len(step_result) == 4:
                        print(f"check gym package! The step_result length is {len(step_result)}")
                        next_observation, reward, done, info = step_result
                    elif isinstance(step_result, tuple) and len(step_result) == 5:
                        print(f"check gym package! The step_result length is {len(step_result)}")
                        next_observation, reward, terminated, truncated, info = step_result
                        done = bool(terminated or truncated)
                    else:
                        raise ValueError(f"Unexpected env.step() return format: {step_result}")
                    print(f"!!!!!!! now in env.step step: {step_idx} done")
                    flag_get_env_response = True
                    break
                except asyncio.TimeoutError:
                    print(f"Attention!!! Timeout occurred while retrieving environment response! This is attempt {retry_num_get_env_response+1} of {self.env_response_retry_num}.")
                    continue
                    
            if not flag_get_env_response:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n", "red")
                reward = 0

                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            delta_time = time.time() - start_time
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len

            # Update agent internal state.
            print(f"now updating agent from env step: {step_idx}")
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )
            if current_compression_event is not None:
                compressed_prompt_text = self.chat_parser.parse(
                    agent.chat_completions.copy(),
                    add_generation_prompt=True,
                    is_first_msg=True,
                )
                current_compression_event["working_prompt_after"] = len(
                    self.tokenizer.encode(
                        compressed_prompt_text, add_special_tokens=False
                    )
                )
            print(f"now updating agent from env step: {step_idx} done")
            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            # Check and convert to tokens if necessary
            assert assistant_message is not None or mode != "Token", "Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assert env_messages is not None or mode != "Token", "Environment messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assistant_msg_tokens = canonical_assistant_tokens
            assistant_msg_masks = canonical_assistant_masks
            env_msg_tokens, env_msg_masks = [], []
            try:
                if env_messages:
                    env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=True)
            except:
                termination_reason = "ENCODE_ERROR"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done

                assistant_message_list = [assistant_message]
                env_messages_list = [env_messages]
                print(f"Ops!!!the assistant_message_list is: {assistant_message_list}")
                print(f"Ops!!!the env_messages_list is: {env_messages_list}")
                break

            if segmentation_enabled and active_policy_segment is not None:
                # Defer observation insertion until a subsequent model call is
                # actually made. Terminal observations therefore remain in
                # Trajectory.steps but do not add unused conditioning tokens.
                pending_segment_observation_tokens = list(env_msg_tokens)

            # Update repsonse token length
            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
            # Reached maximum number of tokens for the trajectory
            if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
                # Truncation length
                truncation_length = self.max_response_length - response_token_len
                # Truncate the response and masks
                if truncation_length < 0:
                    truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                    truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                else:
                    # Edge case where the response is exactly the max response length.
                    truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
                    truncated_response_masks = assistant_msg_masks + env_msg_masks
                # Update token collections
                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)

                model_gen_tokens_len += len(assistant_msg_tokens) 
                env_tokens_len += len(env_msg_tokens)

                cur_step = agent.get_current_state()
                if response_token_len - len(env_msg_tokens) > self.max_response_length:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                # handle returning
                break

            # Update the token version of trajectory
            response_tokens.extend(assistant_msg_tokens)
            model_gen_tokens_len += len(assistant_msg_tokens)

            response_masks.extend(assistant_msg_masks)
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TRAJ_TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            # Check if episode is done
            if done:
                termination_reason = "ENV_DONE"
                break

            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)
            env_tokens_len += len(env_msg_tokens)

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        # end loop, process this single trajectory
        masked_out = False
        reward_half = False
        if self.overlong_filter:
            # if termination_reason == "TRUNCATION" or termination_reason == "MAX_STEPS" or termination_reason == "TIMEOUT" or termination_reason == "NO_CONTAINER":
            if  termination_reason == "NO_CONTAINER" or termination_reason == "PROMPT_TRUNCATION" or termination_reason == "ENCODE_ERROR":
                
                # if not process_terminated:
                # Mask out the entire response for overlong trajectories if the reward is 0.
                response_masks = [0] * len(response_masks)
                masked_out = True

            if termination_reason == "TRUNCATION" or termination_reason =="MAX_STEPS" or termination_reason == "TRAJ_TIMEOUT":
                reward_half = True
                # masked_out = True
                # reward = 0.0

        if hasattr(env, "compute_final_reward") and not masked_out and agent.trajectory.steps:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            if reward_half:
                reward = reward * 0.5
            cur_step.reward = reward

        # Closing environment using the executor.
        print(f"now closing env: {idx}")
        await loop.run_in_executor(self.executor, env.close)
        print(f"now closing env: {idx} done")

        if termination_reason:
            if reward > 0:
                color = "green"
            else:
                color = "yellow"
            print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. "
                f"steps={len(agent.trajectory.steps)}, prompt_tokens={len(prompt_tokens)}, "
                f"response_tokens={len(response_tokens)}, model_gen_tokens={model_gen_tokens_len}.\n",
            )
            if masked_out:
                if process_terminated:
                    print(f"Trajectory {idx} has problems, but is processed into this batch.")
                else:
                    print(f"Trajectory {idx} is masked out due to overlong filter.")

        trajectory: Trajectory = agent.trajectory
        if segmentation_enabled:
            # Do not append a pending terminal observation: no later policy
            # token was conditioned on it. A segment is created only by a real
            # model call, so compression at the end cannot create an empty one.
            pending_segment_observation_tokens = []
            finalize_policy_segment()
        # Aggregate final trajectory statistics
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

        if mode == "Text":
            return trajectory
        elif mode == "Token":
            for compression_event in compression_events:
                compression_event["reward"] = trajectory.reward
                compression_event["termination_reason"] = termination_reason or ""
            for force_candidate in forced_compression_candidates:
                force_candidate["reward"] = trajectory.reward
                force_candidate["termination_reason"] = termination_reason or ""
            token_result = {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "chat_completions": agent.chat_completions,
                # Preserve exactly what the chat parser renders for the next model call.
                "rendered_prompt": self.chat_parser.parse(
                    agent.chat_completions.copy(), add_generation_prompt=True, is_first_msg=True
                ),
                "chat_template_parser": type(self.chat_parser).__name__,
                "termination_reason":termination_reason if termination_reason else "",
                "metrics": {
                    # Total number of steps taken in the trajectory
                    "steps": len(trajectory.steps),
                    # Time to calculate reward
                    "reward_time": reward_time,
                    # Total time spent in environment execution (env.step)
                    "env_time": env_time,
                    # Time to calculate response tokens
                    "llm_time": llm_time,
                    # Total time spent in the trajectory
                    "total_time": total_time,
                    "total_prompt_tokens_num":len(prompt_tokens),
                    "total_response_tokens_num":len(response_tokens),
                    "response_tokens_env_num":env_tokens_len,
                    "response_tokens_model_gen_num":model_gen_tokens_len,
                    "autonomous_compression_calls": autonomous_compression_calls,
                    "forced_compression_calls": forced_compression_calls,
                    "episodes_with_autonomous_compression": int(autonomous_compression_calls > 0),
                    "episodes_with_forced_compression": int(forced_compression_calls > 0),
                    "autonomous_first_compression_position": autonomous_first_compression_position,
                    "forced_first_compression_position": forced_first_compression_position,
                    "forced_compression_retry_count": len(forced_compression_retries),
                    "forced_compression_failed": forced_compression_failed,
                    "forced_compression_skipped_insufficient_budget": forced_compression_skipped_insufficient_budget,
                },
            }
            if segmentation_enabled:
                token_result["episode_id"] = application_id
                token_result["policy_segments"] = [asdict(segment) for segment in policy_segments]
                token_result["forced_compression_retries"] = forced_compression_retries
                token_result["forced_compression_candidates"] = forced_compression_candidates
                token_result["compression_events"] = compression_events
            return token_result
        elif mode == "Conversation":
            return agent.chat_completions
        elif mode == "Step":
            steps_result = {
                "steps": episode_steps,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
            }
            return steps_result

    async def run_agent_trajectory_with_retry(self, idx, application_id, seed=0, mode="Text", **kwargs):
        for _ in range(self.retry_limit):
            try:
                return await asyncio.wait_for(self.run_agent_trajectory_async(idx, application_id=application_id, seed=seed, mode=mode, **kwargs), timeout=7200)
            except Exception:
                traceback.print_exc()
                continue
        traceback.print_exc()
        raise Exception(f"Trajectory {idx} cannot complete. Please check the log message")

    async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", **kwargs):
        if timing_raw is None:
            timing_raw = {}
        assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
        assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"  # type: ignore
        max_concurrency = self.n_parallel_agents
        logger.info(f"in trajectory_generator max_concurrency:{max_concurrency}")

        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

        if self.engine_name == "verl":
            self.rollout_engine.wake_up()

        async def launch_one_trajectory_task(env_idx: int):
            try:
                application_id = str(uuid.uuid4())
                result = await self.run_agent_trajectory_with_retry(
                    idx=env_idx,
                    application_id=application_id,
                    seed=reset_seed,
                    mode=mode,
                    **kwargs,
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                raise e
            return result

        # Create all N conceptual tasks. Their execution will be throttled by the semaphore
        # and the availability of agent/env indices.
        tasks_to_run = [launch_one_trajectory_task(i) for i in range(len(self.envs))]

        tasks_completed = 0
        for coro in asyncio.as_completed(tasks_to_run):
            try:
                result = await coro
                tasks_completed += 1
                colorful_print(f"Number of Trajectories {tasks_completed}/{len(self.envs)} completed", "cyan")
                yield result
            except Exception as e:
                raise e

        if self.engine_name == "verl":
            self.rollout_engine.sleep()

        self.executor.shutdown(wait=False, cancel_futures=True)

    async def execute_tasks(self, tasks: list[dict]):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.

        Args:
            tasks: List of tasks to process
            max_concurrent: Maximum number of concurrent tasks to process (defaults to self.n_parallel_agents)

        Returns:
            A list of trajectories, one for each task.
        """

        max_concurrent = self.n_parallel_agents

        # Initialize results list to store trajectories for all tasks
        all_trajectories = {}

        # Create a queue of tasks to process
        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        # Track completed trajectories
        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with semaphore:
                # Get an available index
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                    self.agents[index] = self.agent_class(**self.agent_args)
                    assert self.agents[index] is not None and isinstance(self.agents[index], BaseAgent), "Agent is not initalized or not inheriting from BaseAgent"
                    self.agents[index].trajectory.task = task  # type: ignore
                    res = await self.run_agent_trajectory_async(index, application_id=task_id)
                    res.task = task
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, res
                finally:
                    # Put the index back in the queue when done
                    await index_queue.put(index)

        # Run all tasks concurrently
        results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])

        all_trajectories = {task_id: trajectory for task_id, trajectory in results}
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
        return ordered_trajectories

    def _convert_prompt_verl(self, prompts, **kwargs):
        """
        Given a list of prompts in Chat template, convert to DataProto format in veRL

        Args:
            prompts: List of prompts to convert
            **kwargs: Additional arguments

        Returns:
            DataProto object containing the converted prompts
        """
        from verl.protocol import DataProto, union_two_dict
        from verl.utils.model import compute_position_id_with_mask
        from verl.utils.torch_functional import pad_sequence_to_length

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        formatted_prompts = [self.chat_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True) for prompt in prompts]

        # Tokenize the final processed strings
        inputs = self.tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        self.tokenizer.padding_side = old_padding_side

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # pad to max sizes
        input_ids = pad_sequence_to_length(input_ids, max_seq_len=self.max_prompt_length, pad_token_id=self.tokenizer.pad_token_id, left_pad=True)
        attention_mask = pad_sequence_to_length(attention_mask, max_seq_len=self.max_prompt_length, pad_token_id=0, left_pad=True)
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)
        data.non_tensor_batch["formatted_prompts"] = np.array(formatted_prompts)

        # original_batch contains the extra info needed for generation
        if "meta_info" in kwargs and kwargs["meta_info"]:
            meta_info = kwargs["meta_info"]
            # only use the original_batch's meta_info since tensor_batch is from batch_dict and non_tensor_batch is not neeeded
            data.meta_info = union_two_dict(data.meta_info, meta_info)

        return data


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
