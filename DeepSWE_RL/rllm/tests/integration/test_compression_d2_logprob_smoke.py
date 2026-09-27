"""Standalone 8-GPU D2/SP2 real-Qwen old-logprob smoke."""

import json
import os

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from rllm.trainer.verl.agent_ppo_trainer import (
    EpisodeRecord,
    _assign_episode_buckets_to_dp,
    _transform_policy_segments_d2,
)
from verl import DataProto
import verl.workers.actor.dp_actor as dp_actor_module
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


MODEL_PATH = os.environ.get(
    "QWEN3_MODEL_PATH", "/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B"
)


def marker(rank, dp, sp, name, **extra):
    print(json.dumps({"marker": name, "rank": rank, "dp": dp, "sp": sp, **extra}), flush=True)


def nonfinite_counts(tensor, mask=None):
    if mask is not None:
        tensor = tensor[mask]
    return {
        "count": int(tensor.numel()),
        "nan": int(torch.isnan(tensor).sum().item()),
        "posinf": int(torch.isposinf(tensor).sum().item()),
        "neginf": int(torch.isneginf(tensor).sum().item()),
    }


def row_fingerprints(data):
    keys = (
        "episode_id", "segment_id", "optimizer_step_id",
        "target_dp_replica", "is_dummy",
    )
    return [
        [
            value.item() if isinstance(value, np.generic) else value
            for value in (data.non_tensor_batch[key][i] for key in keys)
        ]
        for i in range(len(data))
    ]


def parameter_nonfinite_counts(module):
    result = {"elements": 0, "nan": 0, "posinf": 0, "neginf": 0, "max_abs": 0.0}
    for parameter in module.parameters():
        tensor = parameter.detach()
        result["elements"] += tensor.numel()
        result["nan"] += int(torch.isnan(tensor).sum())
        result["posinf"] += int(torch.isposinf(tensor).sum())
        result["neginf"] += int(torch.isneginf(tensor).sum())
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel():
            result["max_abs"] = max(result["max_abs"], float(finite.abs().max()))
    return result


def segment(segment_id, prompt_len, response_len, token_offset):
    return {
        "segment_id": segment_id,
        "prompt_tokens": [100 + (token_offset + i) % 1000 for i in range(prompt_len)],
        "response_tokens": [1200 + (token_offset + i) % 1000 for i in range(response_len)],
        "policy_mask": [1 if i % 3 != 1 else 0 for i in range(response_len)],
    }


def episode(source, step, specs):
    return EpisodeRecord(
        source_batch_idx=source,
        episode_id=f"step{step}-episode{source}",
        group_uid=f"step{step}-group{source}",
        reward=float(source % 2),
        termination_reason="done",
        policy_segments=[
            segment(index, p, r, source * 100 + index * 10)
            for index, (p, r) in enumerate(specs)
        ],
        scalar_advantage=0.25 if source % 2 == 0 else -0.25,
    )


def build_fixture(eos_token_id, pad_token_id, effective_dp_degree, sp_size):
    step0_specs = [
        [(15, 7), (15, 7), (15, 7)],
        [(25, 8), (25, 8)],
        [(45, 9)],
        [(10, 5)],
        [(20, 6)],
    ]
    step1_specs = [
        [(12, 5)],
        [(30, 7), (18, 5), (16, 5)],
        [(38, 8), (22, 6)],
        [(9, 4)],
        [(24, 7)],
    ]
    buckets, source = [], 0
    for step, all_specs in enumerate((step0_specs, step1_specs)):
        bucket = []
        for specs in all_specs:
            bucket.append(episode(source, step, specs))
            source += 1
        buckets.append(bucket)
    episodes = [item for bucket in buckets for item in bucket]
    optimizer_step_by_source = {
        item.source_batch_idx: step
        for step, bucket in enumerate(buckets)
        for item in bucket
    }
    assignment = _assign_episode_buckets_to_dp(buckets, effective_dp_degree)
    batch = _transform_policy_segments_d2(
        episodes,
        optimizer_step_by_source=optimizer_step_by_source,
        assignment=assignment,
        effective_dp_degree=effective_dp_degree,
        sp_size=sp_size,
        eos_token_id=eos_token_id,
        max_prompt_length=128,
        max_response_length=64,
        pad_token_id=pad_token_id,
    )
    batch.meta_info.update(
        {
            "temperature": 1.0,
            "multi_turn": False,
            "global_token_num": batch.batch["attention_mask"].sum(-1).tolist(),
        }
    )
    return episodes, buckets, assignment, batch


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    sp_size = int(os.environ.get("DIAGNOSTIC_SP", "2"))
    assert world % sp_size == 0
    effective_d = world // sp_size
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    worker_config = OmegaConf.create(
        {
            "model": {"lora_rank": 0},
            "actor": {
                "fsdp_config": {"fsdp_size": -1},
                "ulysses_sequence_parallel_size": sp_size,
                "ppo_mini_batch_size": 16,
                "ppo_micro_batch_size": None,
                "ppo_micro_batch_size_per_gpu": 1,
            },
            "rollout": {
                "n": 1,
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 256,
                "log_prob_use_dynamic_bsz": False,
                "temperature": 1.0,
            },
            "ref": {"log_prob_micro_batch_size": None},
        }
    )
    worker = ActorRolloutRefWorker(worker_config, role="actor")
    coordinate = worker.ulysses_device_mesh.get_coordinate()
    dp, sp = int(coordinate[0]), int(coordinate[1])
    assert (dp, sp) == (rank // sp_size, rank % sp_size)
    marker(rank, dp, sp, "mesh-ready")
    assert worker.world_size == dist.get_world_size() == 8
    assert worker.rank == dist.get_rank() == rank
    marker(
        rank,
        dp,
        sp,
        "worker-ready",
        worker_world_size=worker.world_size,
        worker_rank=worker.rank,
        worker_dp=dp,
        worker_sp=sp,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    episodes, buckets, assignment, global_batch = build_fixture(
        tokenizer.eos_token_id, tokenizer.pad_token_id or tokenizer.eos_token_id,
        effective_d, sp_size,
    )
    keep_original = os.environ.get("KEEP_ORIGINAL_DPS")
    if keep_original is not None:
        keep_dps = {int(value) for value in keep_original.split(",") if value}
        dummy_indices = np.flatnonzero(global_batch.non_tensor_batch["is_dummy"])
        assert len(dummy_indices)
        template = int(dummy_indices[0])
        for row, target in enumerate(global_batch.non_tensor_batch["target_dp_replica"]):
            if int(target) not in keep_dps:
                for key in global_batch.batch.keys():
                    global_batch.batch[key][row].copy_(global_batch.batch[key][template])
    prefix_n = os.environ.get("DP0_PREFIX_N")
    if prefix_n is not None:
        prefix_n = int(prefix_n)
        # DP0 logical preprocessing order begins with global rows 0..3:
        # call0, call1, call2, bad call3. Keep only the requested immediate
        # predecessors and the bad call; make every other row dummy-only.
        keep_rows = {3} | set(range(max(0, 3 - prefix_n), 3))
        dummy_indices = np.flatnonzero(global_batch.non_tensor_batch["is_dummy"])
        template = int(dummy_indices[0])
        for row in range(len(global_batch)):
            if row not in keep_rows:
                for key in global_batch.batch.keys():
                    global_batch.batch[key][row].copy_(global_batch.batch[key][template])
    if os.environ.get("SYNTHETIC_BAD") == "1":
        row = 3
        attended = global_batch.batch["attention_mask"][row].bool()
        global_batch.batch["input_ids"][row][attended] = 710
        response_width = global_batch.batch["responses"].size(1)
        response_attention = global_batch.batch["attention_mask"][row, -response_width:].bool()
        global_batch.batch["responses"][row][response_attention] = 710
    assert len(global_batch) % world == 0
    rows_per_rank = len(global_batch) // world
    local_batch = global_batch.chunk(world)[rank]
    local_targets = local_batch.non_tensor_batch["target_dp_replica"]
    assert np.all(local_targets == dp)
    local_layout = [
        {
            key: (
                value.item() if isinstance(value, np.generic) else value
            )
            for key, value in {
                "optimizer_step_id": local_batch.non_tensor_batch["optimizer_step_id"][i],
                "target_dp_replica": local_batch.non_tensor_batch["target_dp_replica"][i],
                "is_dummy": local_batch.non_tensor_batch["is_dummy"][i],
                "episode_id": local_batch.non_tensor_batch["episode_id"][i],
                "segment_id": local_batch.non_tensor_batch["segment_id"][i],
            }.items()
        }
        for i in range(len(local_batch))
    ]
    marker(rank, dp, sp, "layout-ready", rows=len(local_batch), layout=local_layout)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        attn_implementation=os.environ.get("ATTN_IMPL", "flash_attention_2"), trust_remote_code=True,
    )
    model = FSDP(model, device_id=device, use_orig_params=False)
    d2_bad_state = {
        "active": False,
        "reports": [],
        "layer_reports": [],
        "submodule_reports": [],
    }

    def tensor_from_output(output):
        return output[0] if isinstance(output, tuple) else output

    def make_layer_hook(layer_index):
        def hook(module, inputs, output):
            if not d2_bad_state["active"]:
                return
            tensor = tensor_from_output(output).detach()
            flat = tensor.reshape(-1, tensor.size(-1))
            nan_tokens = torch.isnan(flat).any(dim=-1).nonzero(as_tuple=False).flatten()
            finite = tensor[torch.isfinite(tensor)]
            d2_bad_state["layer_reports"].append(
                {
                    "rank": rank,
                    "layer": layer_index,
                    "nan_elements": int(torch.isnan(tensor).sum()),
                    "inf_elements": int(torch.isinf(tensor).sum()),
                    "nan_local_token_indices": nan_tokens.tolist(),
                    "max_abs_finite": float(finite.abs().max()) if finite.numel() else None,
                }
            )
        return hook

    def make_submodule_hook(layer_index, name):
        def hook(module, inputs, output):
            if not d2_bad_state["active"]:
                return
            tensor = tensor_from_output(output).detach()
            input_tensor = tensor_from_output(inputs).detach()
            finite = tensor[torch.isfinite(tensor)]
            input_finite = input_tensor[torch.isfinite(input_tensor)]
            parameter = next(module.parameters(), None)
            save_tag = os.environ.get("QKV_SAVE_TAG")
            if (
                save_tag
                and rank == 1
                and layer_index == 0
                and name in {"q_proj", "k_proj", "v_proj"}
            ):
                torch.save(
                    tensor.detach().cpu(),
                    f"/tmp/{save_tag}_{name}.pt",
                )
            d2_bad_state["submodule_reports"].append(
                {
                    "rank": rank,
                    "layer": layer_index,
                    "name": name,
                    "nan_elements": int(torch.isnan(tensor).sum()),
                    "inf_elements": int(torch.isinf(tensor).sum()),
                    "max_abs_finite": float(finite.abs().max()) if finite.numel() else None,
                    "input_nan_elements": int(torch.isnan(input_tensor).sum()),
                    "input_inf_elements": int(torch.isinf(input_tensor).sum()),
                    "input_max_abs_finite": float(input_finite.abs().max()) if input_finite.numel() else None,
                    "parameter_nan_elements": int(torch.isnan(parameter).sum()) if parameter is not None else None,
                    "parameter_inf_elements": int(torch.isinf(parameter).sum()) if parameter is not None else None,
                }
            )
        return hook

    diagnostic_handles = []
    for layer_index, layer in enumerate(model.module.model.layers):
        diagnostic_handles.append(layer.register_forward_hook(make_layer_hook(layer_index)))
        modules = {
            "input_layernorm": layer.input_layernorm,
            "q_proj": layer.self_attn.q_proj,
            "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj,
            "o_proj": layer.self_attn.o_proj,
            "post_attention_layernorm": layer.post_attention_layernorm,
            "gate_proj": layer.mlp.gate_proj,
            "up_proj": layer.mlp.up_proj,
            "down_proj": layer.mlp.down_proj,
        }
        for name, module in modules.items():
            diagnostic_handles.append(
                module.register_forward_hook(make_submodule_hook(layer_index, name))
            )

    def d2_bad_lm_head_hook(module, inputs, output):
        if not d2_bad_state["active"]:
            if os.environ.get("SYNC_MODEL_FORWARD") == "1":
                torch.cuda.synchronize()
            return
        hidden = inputs[0].detach()
        logits = output.detach()
        report = {
            "rank": rank,
            "hidden": nonfinite_counts(hidden),
            "hidden_max_abs": float(hidden.float().abs().max()),
            "hidden_norm": float(hidden.float().norm()),
            "logits": nonfinite_counts(logits),
        }
        padded_total = ((23 + sp_size - 1) // sp_size) * sp_size
        shard_width = padded_total // sp_size
        owner_sp = 17 // shard_width
        local_prediction_index = 17 % shard_width
        if sp == owner_sp:
            vector = logits.reshape(-1, logits.size(-1))[local_prediction_index]
            selected = vector[1810]
            log_z_current = torch.logsumexp(vector, dim=-1)
            log_z_fp32 = torch.logsumexp(vector.float(), dim=-1)
            report.update(
                {
                    "global_prediction_index": 17,
                    "owner_sp_rank": owner_sp,
                    "local_prediction_index": local_prediction_index,
                    "vocab_vector": nonfinite_counts(vector),
                    "selected_raw_logit": float(selected.float()),
                    "selected_isfinite": bool(torch.isfinite(selected)),
                    "log_z_current": float(log_z_current.float()),
                    "log_z_current_isfinite": bool(torch.isfinite(log_z_current)),
                    "log_z_fp32": float(log_z_fp32),
                    "log_z_fp32_isfinite": bool(torch.isfinite(log_z_fp32)),
                    "manual_fp32_logprob": float(selected.float() - log_z_fp32),
                }
            )
        d2_bad_state["reports"].append(report)
        if os.environ.get("SYNC_MODEL_FORWARD") == "1":
            torch.cuda.synchronize()

    d2_bad_hook_handle = model.module.lm_head.register_forward_hook(
        d2_bad_lm_head_hook
    )
    actor_config = OmegaConf.create(
        {
            "use_remove_padding": True,
            "use_fused_kernels": False,
            "ulysses_sequence_parallel_size": sp_size,
            "use_torch_compile": False,
        }
    )
    actor = DataParallelPPOActor(actor_config, model)
    actor_output_reports = []
    bad_aux = {}
    original_compute_log_prob = actor.compute_log_prob

    def recording_compute_log_prob(data, *args, **kwargs):
        matches = (data.batch["responses"][:, 0] == 1810).nonzero(as_tuple=False)
        if len(matches):
            row = int(matches[0, 0])
            for key in ("policy_mask", "actor_token_weight"):
                bad_aux[key] = data.batch[key][row : row + 1].detach().cpu().clone()
        result = original_compute_log_prob(data, *args, **kwargs)
        log_prob = result[0] if isinstance(result, tuple) else result
        actor_output_reports.append(
            {
                "shape": list(log_prob.shape),
                "nonfinite": nonfinite_counts(log_prob),
                "fingerprints": row_fingerprints(data),
            }
        )
        return result

    actor.compute_log_prob = recording_compute_log_prob
    trim_reports = []
    original_trim = dp_actor_module._trim_compression_microbatch

    def recording_trim(micro_batch, prompt_carrier_width, response_carrier_width):
        trimmed, stats = original_trim(
            micro_batch, prompt_carrier_width, response_carrier_width
        )
        d2_bad_state["active"] = bool(
            int(trimmed["responses"][0, 0].item()) == 1810
            and int(stats["prompt_local_width"]) == 18
            and int(stats["response_local_width"]) == 5
        )
        if (
            rank == 0
            and int(trimmed["responses"][0, 0].item()) == 1810
            and int(stats["prompt_local_width"]) == 18
            and int(stats["response_local_width"]) == 5
        ):
            torch.save(
                ({
                    key: trimmed[key].detach().cpu().clone()
                    for key in (
                        "input_ids", "attention_mask", "position_ids", "responses",
                        "policy_mask", "actor_token_weight",
                    )
                    if key in trimmed
                } | bad_aux)
                | {
                    "p_carrier": int(prompt_carrier_width),
                    "r_carrier": int(response_carrier_width),
                    "p_local": int(stats["prompt_local_width"]),
                    "r_local": int(stats["response_local_width"]),
                    "total_nnz": int(stats["total_nnz"]),
                    "sp_pad": int((-stats["total_nnz"]) % sp_size),
                },
                "/tmp/stage8c_d2_bad_microbatch.pt",
            )
        trim_reports.append(
            {
                "p_carrier": int(prompt_carrier_width),
                "r_carrier": int(response_carrier_width),
                "p_local": int(stats["prompt_local_width"]),
                "r_local": int(stats["response_local_width"]),
                "total_nnz": int(stats["total_nnz"]),
                "sp_pad": int((-stats["total_nnz"]) % sp_size),
                "saved_rectangular_elements": int(
                    len(micro_batch)
                    * (prompt_carrier_width + response_carrier_width)
                    - stats["local_capacity"]
                ),
            }
        )
        return trimmed, stats

    dp_actor_module._trim_compression_microbatch = recording_trim
    worker.actor_module_fsdp = model
    worker.actor = actor
    parameters_before = parameter_nonfinite_counts(model)
    torch.cuda.reset_peak_memory_stats()
    marker(rank, dp, sp, "preprocess-ready")
    marker(rank, dp, sp, "forward-start")
    output = worker.compute_compression_log_prob_d2(local_batch)
    parameters_after = parameter_nonfinite_counts(model)
    marker(rank, dp, sp, "forward-done", output_rows=len(output))
    torch.cuda.synchronize()
    memory = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }
    marker(rank, dp, sp, "postprocess-done")

    local_old = output.batch["old_log_probs"].to(device)
    gathered_old = [torch.empty_like(local_old) for _ in range(world)]
    dist.all_gather(gathered_old, local_old)
    global_old = torch.cat(gathered_old, dim=0).cpu()
    per_rank = {
        "rank": rank, "dp": dp, "sp": sp, "rows": len(local_batch),
        "forward_count": len(local_batch) * 2,
        "actor_output_reports": actor_output_reports,
        "d2_bad_layer_reports": d2_bad_state["reports"],
        "transformer_layer_reports": d2_bad_state["layer_reports"],
        "submodule_reports": d2_bad_state["submodule_reports"],
        "postprocess_shape": list(local_old.shape),
        "postprocess_nonfinite": nonfinite_counts(local_old),
        "input_fingerprints": row_fingerprints(local_batch),
        "output_fingerprints": row_fingerprints(local_batch),
        "trim_reports": trim_reports,
        "memory": memory,
        "parameters_before": parameters_before,
        "parameters_after": parameters_after,
    }
    rank_reports = [None] * world
    dist.all_gather_object(rank_reports, per_rank)

    if rank == 0:
        assignment_report = []
        for step, bucket in enumerate(buckets):
            for item in bucket:
                assignment_report.append(
                    {
                        "source": item.source_batch_idx,
                        "step": step,
                        "dp": assignment.target_dp_by_source[item.source_batch_idx],
                        "cost": assignment.episode_cost_by_source[item.source_batch_idx],
                        "segments": len(item.policy_segments),
                    }
                )
        step_reports = {}
        for step in (0, 1):
            step_mask = global_batch.non_tensor_batch["optimizer_step_id"] == step
            real_mask = step_mask & ~global_batch.non_tensor_batch["is_dummy"]
            dummy_mask = step_mask & global_batch.non_tensor_batch["is_dummy"]
            real_per_dp = [
                int(np.sum(real_mask & (global_batch.non_tensor_batch["target_dp_replica"] == dp_idx)))
                for dp_idx in range(effective_d)
            ]
            dummy_per_dp = [
                int(np.sum(dummy_mask & (global_batch.non_tensor_batch["target_dp_replica"] == dp_idx)))
                for dp_idx in range(effective_d)
            ]
            step_reports[str(step)] = {
                "episodes": len(buckets[step]),
                "real_rows_per_dp": real_per_dp,
                "dummy_rows_per_dp": dummy_per_dp,
                "target_rows": real_per_dp[0] + dummy_per_dp[0],
                "real_actor_weight_sum": float(global_batch.batch["actor_token_weight"][real_mask].sum()),
                "dummy_actor_weight_sum": float(global_batch.batch["actor_token_weight"][dummy_mask].sum()),
            }
        response_width = global_batch.batch["responses"].size(1)
        response_attention = global_batch.batch["attention_mask"][:, -response_width:].bool()
        # This is the collector definition that failed in the previous run.
        valid = response_attention
        invalid = ~valid
        policy_mask = global_batch.batch["policy_mask"].bool()
        actor_weight = global_batch.batch["actor_token_weight"]
        advantages = global_batch.batch["advantages"]
        episode_weight = global_batch.batch["episode_normalized_weight"]
        dummy_rows_mask = torch.as_tensor(
            global_batch.non_tensor_batch["is_dummy"], dtype=torch.bool
        ).unsqueeze(1).expand_as(valid)
        real_rows_mask = ~dummy_rows_mask
        training_active = actor_weight > 0
        observation = real_rows_mask & valid & ~policy_mask
        padding = invalid
        categories = {
            "real_policy_weight_gt_0": real_rows_mask & policy_mask & training_active,
            "real_policy_weight_eq_0": real_rows_mask & policy_mask & ~training_active,
            "observation": observation,
            "dummy": dummy_rows_mask & valid,
            "padding": padding,
            "policy_all": policy_mask,
            "training_active": training_active,
        }
        finite_report = {
            "all": nonfinite_counts(global_old),
            "valid": nonfinite_counts(global_old, valid),
            "invalid": nonfinite_counts(global_old, invalid),
            "categories": {
                name: nonfinite_counts(global_old, mask)
                for name, mask in categories.items()
            },
        }
        bad = valid & ~torch.isfinite(global_old)
        bad_coords = bad.nonzero(as_tuple=False)
        bad_details = []
        for row_tensor, col_tensor in bad_coords[:20]:
            row, col = int(row_tensor), int(col_tensor)
            token_id = int(global_batch.batch["responses"][row, col])
            real_response_len = int(response_attention[row].sum())
            bad_details.append(
                {
                    "global_row_index": row,
                    "response_column": col,
                    "old_log_prob": str(global_old[row, col].item()),
                    "episode_id": str(global_batch.non_tensor_batch["episode_id"][row]),
                    "group_uid": str(global_batch.non_tensor_batch["group_uid"][row]),
                    "segment_id": int(global_batch.non_tensor_batch["segment_id"][row]),
                    "source_batch_idx": int(global_batch.non_tensor_batch["source_batch_idx"][row]),
                    "optimizer_step_id": int(global_batch.non_tensor_batch["optimizer_step_id"][row]),
                    "target_dp_replica": int(global_batch.non_tensor_batch["target_dp_replica"][row]),
                    "is_dummy": bool(global_batch.non_tensor_batch["is_dummy"][row]),
                    "response_attention_mask": int(response_attention[row, col]),
                    "policy_mask": int(policy_mask[row, col]),
                    "actor_token_weight": float(actor_weight[row, col]),
                    "advantages": float(advantages[row, col]),
                    "episode_normalized_weight": float(episode_weight[row, col]),
                    "token_id": token_id,
                    "token_string": tokenizer.decode([token_id]),
                    "real_response_len": real_response_len,
                    "column_outside_real_response": col >= real_response_len,
                }
            )
        dummy_details = []
        for row in np.flatnonzero(global_batch.non_tensor_batch["is_dummy"]):
            dummy_details.append(
                {
                    "row": int(row),
                    "response_token_ids": global_batch.batch["responses"][row].tolist(),
                    "response_attention": response_attention[row].to(torch.int).tolist(),
                    "policy_mask": policy_mask[row].to(torch.int).tolist(),
                    "actor_token_weight": actor_weight[row].tolist(),
                    "old_log_probs": [str(value) for value in global_old[row].tolist()],
                }
            )
        collected_fingerprints = [
            fp for rank_report in rank_reports for fp in rank_report["output_fingerprints"]
        ]
        expected_fingerprints = row_fingerprints(global_batch)
        fingerprint_identity = collected_fingerprints == expected_fingerprints
        assert tuple(global_old.shape) == tuple(global_batch.batch["responses"].shape)
        report = {
            "world": world, "sp": sp_size, "d": effective_d,
            "rows_per_rank": rows_per_rank,
            "global_input_rows": len(global_batch),
            "old_log_probs_shape": list(global_old.shape),
            "p_carrier": global_batch.meta_info["compression_prompt_carrier_width"],
            "r_carrier": global_batch.meta_info["compression_response_carrier_width"],
            "collector_valid_definition": "attention_mask[:, -response_width:].bool()",
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "finite_report": finite_report,
            "bad_valid_total": int(bad_coords.size(0)),
            "bad_valid_first_20": bad_details,
            "dummy_details": dummy_details,
            "dummy_rows": int(global_batch.non_tensor_batch["is_dummy"].sum()),
            "assignment": assignment_report,
            "steps": step_reports,
            "ranks": rank_reports,
            "metadata_identity": fingerprint_identity,
        }
        print(json.dumps(report, indent=2), flush=True)
        assert fingerprint_identity
        for step in step_reports.values():
            assert abs(step["real_actor_weight_sum"] - effective_d) < 1e-5
            assert step["dummy_actor_weight_sum"] == 0.0
    dist.barrier()
    marker(rank, dp, sp, "collector-done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
