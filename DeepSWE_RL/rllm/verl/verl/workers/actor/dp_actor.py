# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_policy_loss,
    compute_policy_loss_weighted,
    kl_penalty,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_COMPRESSION_RESPONSE_ALIGNED_FIELDS = (
    "responses",
    "advantages",
    "policy_mask",
    "episode_normalized_weight",
    "actor_token_weight",
    "old_log_probs",
    "ref_log_prob",
)


def _trim_compression_microbatch(micro_batch, prompt_carrier_width, response_carrier_width):
    """Trim one rectangular compression carrier to its local P/R widths.

    Attention, rather than policy_mask, defines real response tokens because
    tool observations are conditioning tokens with policy_mask == 0.
    """
    values = {key: micro_batch[key] for key in micro_batch.keys()}
    attention_mask = values["attention_mask"]
    if attention_mask.dim() != 2:
        raise ValueError("Compression minimal padding requires a 2-D attention_mask")
    expected_width = prompt_carrier_width + response_carrier_width
    if attention_mask.size(1) != expected_width:
        raise ValueError(
            "Compression carrier width mismatch: "
            f"attention_width={attention_mask.size(1)}, expected={expected_width}"
        )

    prompt_attention = attention_mask[:, :prompt_carrier_width]
    response_attention = attention_mask[
        :, prompt_carrier_width : prompt_carrier_width + response_carrier_width
    ]
    prompt_lengths = prompt_attention.sum(dim=-1).to(torch.long)
    response_lengths = response_attention.sum(dim=-1).to(torch.long)
    if torch.any(prompt_lengths <= 0) or torch.any(response_lengths <= 0):
        raise ValueError("Compression segment rows must contain prompt and response tokens")

    prompt_columns = torch.arange(
        prompt_carrier_width, device=attention_mask.device
    ).unsqueeze(0)
    expected_prompt = prompt_columns >= (
        prompt_carrier_width - prompt_lengths
    ).unsqueeze(1)
    response_columns = torch.arange(
        response_carrier_width, device=attention_mask.device
    ).unsqueeze(0)
    expected_response = response_columns < response_lengths.unsqueeze(1)
    if not torch.equal(prompt_attention.bool(), expected_prompt):
        raise ValueError(
            "Compression prompt attention mask must be contiguous left padding: 0...0 1...1"
        )
    if not torch.equal(response_attention.bool(), expected_response):
        raise ValueError(
            "Compression response attention mask must be contiguous right padding: 1...1 0...0"
        )

    prompt_local_width = int(prompt_lengths.max().item())
    response_local_width = int(response_lengths.max().item())
    prompt_start = prompt_carrier_width - prompt_local_width
    prompt_slice = slice(prompt_start, prompt_carrier_width)
    response_input_slice = slice(
        prompt_carrier_width, prompt_carrier_width + response_local_width
    )

    for key in ("input_ids", "attention_mask", "position_ids"):
        if key not in values:
            continue
        tensor = values[key]
        if tensor.dim() != 2:
            raise ValueError(
                f"Compression minimal padding requires 2-D {key}, got {tensor.dim()} dimensions"
            )
        values[key] = torch.cat(
            [tensor[:, prompt_slice], tensor[:, response_input_slice]], dim=-1
        )
    if "prompts" in values:
        values["prompts"] = values["prompts"][:, prompt_slice]
    for key in _COMPRESSION_RESPONSE_ALIGNED_FIELDS:
        if key in values:
            if values[key].size(-1) != response_carrier_width:
                raise ValueError(
                    f"Compression response-aligned field {key!r} has width "
                    f"{values[key].size(-1)}, expected {response_carrier_width}"
                )
            values[key] = values[key][:, :response_local_width]

    total_nnz = int(values["attention_mask"].sum().item())
    return values, {
        "prompt_local_width": prompt_local_width,
        "response_local_width": response_local_width,
        "total_local_width": prompt_local_width + response_local_width,
        "total_nnz": total_nnz,
        "local_capacity": len(attention_mask)
        * (prompt_local_width + response_local_width),
    }


def _restore_compression_response_width(tensor, response_carrier_width):
    """Right-pad a local logprob/entropy result back to the carrier API."""
    local_width = tensor.size(-1)
    if local_width > response_carrier_width:
        raise ValueError(
            f"Compression local response width {local_width} exceeds carrier width "
            f"{response_carrier_width}"
        )
    if local_width == response_carrier_width:
        return tensor
    return torch.nn.functional.pad(
        tensor, (0, response_carrier_width - local_width), value=0.0
    )


def _compression_carrier_widths(data):
    """Read real carrier metadata, with a centralized legacy-test fallback."""
    response_width = data.meta_info.get("compression_response_carrier_width")
    prompt_width = data.meta_info.get("compression_prompt_carrier_width")
    if response_width is None:
        response_width = data.batch["responses"].size(-1)
    if prompt_width is None:
        prompt_width = data.batch["input_ids"].size(-1) - int(response_width)
    if int(prompt_width) <= 0 or int(response_width) <= 0:
        raise ValueError("Compression carrier widths must both be positive")
    return int(prompt_width), int(response_width)


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        compression_segment_training = data.meta_info.get(
            "compression_segment_training", False
        )
        prompt_carrier_width = response_carrier_width = None
        if compression_segment_training:
            prompt_carrier_width, response_carrier_width = (
                _compression_carrier_widths(data)
            )
            if getattr(self, "use_ulysses_sp", False) and not getattr(
                self, "use_remove_padding", False
            ):
                raise RuntimeError(
                    "compression minimal padding with Ulysses SP currently requires "
                    "use_remove_padding=True"
                )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            if compression_segment_training:
                micro_batch, _ = _trim_compression_microbatch(
                    micro_batch,
                    int(prompt_carrier_width),
                    int(response_carrier_width),
                )
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            if compression_segment_training:
                log_probs = _restore_compression_response_width(
                    log_probs, int(response_carrier_width)
                )
                if calculate_entropy:
                    entropy = _restore_compression_response_width(
                        entropy, int(response_carrier_width)
                    )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        compression_segment_training = data.meta_info.get(
            "compression_segment_training", False
        )
        prompt_carrier_width = response_carrier_width = None

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if compression_segment_training:
            prompt_carrier_width, response_carrier_width = (
                _compression_carrier_widths(data)
            )
            if getattr(self, "use_ulysses_sp", False) and not getattr(
                self, "use_remove_padding", False
            ):
                raise RuntimeError(
                    "compression minimal padding with Ulysses SP currently requires "
                    "use_remove_padding=True"
                )
            required_fields = set(select_keys + ["policy_mask", "actor_token_weight"])
            if self.config.use_kl_loss:
                required_fields.add("ref_log_prob")
            missing_fields = sorted(required_fields - set(data.batch.keys()))
            if missing_fields:
                raise ValueError(
                    "Compression segment training batch is missing required fields: "
                    f"{missing_fields}"
                )
            select_keys.extend(["policy_mask", "actor_token_weight"])
            if self.config.use_dynamic_bsz:
                raise RuntimeError(
                    "compression segment training stage 7C-D1 currently requires use_dynamic_bsz=False"
                )
            if "optimizer_step_id" not in data.non_tensor_batch:
                raise ValueError(
                    "Compression segment training batch is missing optimizer_step_id"
                )
        elif multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if compression_segment_training:
            optimizer_step_ids = data.non_tensor_batch["optimizer_step_id"]
            unique_step_ids = sorted({int(value) for value in optimizer_step_ids})
            if unique_step_ids != list(range(len(unique_step_ids))):
                raise ValueError(
                    "Compression optimizer_step_id values must be contiguous from zero"
                )
            dataloader = [
                batch[np.where(optimizer_step_ids == step_id)[0]]
                for step_id in unique_step_ids
            ]
            if any(len(step_batch) == 0 for step_batch in dataloader):
                raise ValueError("Compression optimizer step cannot be empty")
        elif has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        compression_padding_stats = {
            "local_capacity": 0,
            "sp_padding": 0,
            "max_prompt": 0,
            "max_response": 0,
            "max_total": 0,
        }
        if compression_segment_training:
            real_tokens = int(batch["attention_mask"].sum().item())
            row_count = len(batch)
            hard_capacity = row_count * (
                int(data.meta_info.get("compression_max_prompt_length", prompt_carrier_width))
                + int(data.meta_info.get("compression_max_response_length", response_carrier_width))
            )
            carrier_capacity = row_count * (
                int(prompt_carrier_width) + int(response_carrier_width)
            )
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                weighted_step_metrics = {}

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    if compression_segment_training:
                        data, trim_stats = _trim_compression_microbatch(
                            data,
                            int(prompt_carrier_width),
                            int(response_carrier_width),
                        )
                        if epoch == 0:
                            compression_padding_stats["local_capacity"] += trim_stats[
                                "local_capacity"
                            ]
                            compression_padding_stats["sp_padding"] += (
                                self.ulysses_sequence_parallel_size
                                - trim_stats["total_nnz"]
                                % self.ulysses_sequence_parallel_size
                            ) % self.ulysses_sequence_parallel_size
                            compression_padding_stats["max_prompt"] = max(
                                compression_padding_stats["max_prompt"],
                                trim_stats["prompt_local_width"],
                            )
                            compression_padding_stats["max_response"] = max(
                                compression_padding_stats["max_response"],
                                trim_stats["response_local_width"],
                            )
                            compression_padding_stats["max_total"] = max(
                                compression_padding_stats["max_total"],
                                trim_stats["total_local_width"],
                            )
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if compression_segment_training:
                        response_mask = data["policy_mask"]
                    elif multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    if compression_segment_training:
                        actor_token_weight = data["actor_token_weight"]
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = (
                            compute_policy_loss_weighted(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                policy_mask=response_mask,
                                actor_token_weight=actor_token_weight,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                            )
                        )
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    if entropy_coeff != 0:
                        if compression_segment_training:
                            entropy_loss = torch.sum(entropy * actor_token_weight)
                        else:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        if compression_segment_training:
                            kl_loss = torch.sum(kld * actor_token_weight)
                        else:
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        if not compression_segment_training:
                            metrics["actor/kl_loss"] = kl_loss.detach().item()
                            metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if compression_segment_training:
                        # actor_token_weight already carries the complete optimizer-step
                        # normalization. Microbatch gradients must add without rescaling.
                        loss = policy_loss
                    elif self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    if compression_segment_training:
                        if entropy_coeff != 0:
                            micro_metrics["actor/entropy_loss"] = entropy_loss.detach().item()
                        if self.config.use_kl_loss:
                            micro_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        for key, value in micro_metrics.items():
                            weighted_step_metrics[key] = weighted_step_metrics.get(key, 0.0) + value
                    else:
                        append_to_dict(metrics, micro_metrics)

                if compression_segment_training:
                    if self.config.use_kl_loss:
                        weighted_step_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    append_to_dict(metrics, weighted_step_metrics)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        if compression_segment_training:
            local_capacity = compression_padding_stats["local_capacity"]
            append_to_dict(
                metrics,
                {
                    "compression/padding_tokens_before": hard_capacity,
                    "compression/padding_tokens_after": local_capacity,
                    "compression/padding_tokens_saved": hard_capacity
                    - local_capacity,
                    "compression/padding_ratio_before": (
                        (hard_capacity - real_tokens) / hard_capacity
                        if hard_capacity
                        else 0.0
                    ),
                    "compression/padding_ratio_after": (
                        (local_capacity - real_tokens) / local_capacity
                        if local_capacity
                        else 0.0
                    ),
                    "compression/batch_padding_tokens": carrier_capacity
                    - real_tokens,
                    "compression/sp_divisibility_padding_tokens": compression_padding_stats[
                        "sp_padding"
                    ],
                    "compression/max_microbatch_prompt_len": compression_padding_stats[
                        "max_prompt"
                    ],
                    "compression/max_microbatch_response_len": compression_padding_stats[
                        "max_response"
                    ],
                    "compression/max_microbatch_total_len": compression_padding_stats[
                        "max_total"
                    ],
                },
            )
        self.actor_optimizer.zero_grad()
        return metrics
