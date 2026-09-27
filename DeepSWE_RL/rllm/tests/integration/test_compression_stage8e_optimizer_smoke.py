"""One-step real-Qwen production-worker compression D2 optimizer smoke."""

import json
import os
import time
import copy

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers.integrations import flash_attention as flash_attention_integration

from tests.integration.test_compression_d2_logprob_smoke import build_fixture
from verl.models.transformers import monkey_patch as ulysses_patch_module
import verl.workers.actor.dp_actor as dp_actor_module
from verl.workers.fsdp_workers import ActorRolloutRefWorker


ROOT = "/home/yyk/yyk11/zhongtianyang/memory/SWE-Master/DeepSWE_RL/rllm"
MODEL = "/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B"
CONFIG = f"{ROOT}/rllm/trainer/config/ppo_trainer.yaml"


def marker(rank, name, **extra):
    print(json.dumps({"marker": name, "rank": rank, **extra}), flush=True)


def finite_counts(tensor, mask=None):
    if mask is not None:
        mask = mask.to(device=tensor.device, dtype=torch.bool)
        tensor = tensor[mask]
    return {
        "count": int(tensor.numel()),
        "nan": int(torch.isnan(tensor).sum()),
        "posinf": int(torch.isposinf(tensor).sum()),
        "neginf": int(torch.isneginf(tensor).sum()),
        "nonzero": int((tensor != 0).sum()),
        "max_abs": float(tensor.detach().abs().max().float().cpu()) if tensor.numel() else 0.0,
    }


def clone_dataproto(data):
    """Clone a physical-row DataProto before an in-place D2 RPC preprocess."""
    return type(data)(
        batch=data.batch.clone() if data.batch is not None else None,
        non_tensor_batch={key: value.copy() for key, value in data.non_tensor_batch.items()},
        meta_info=copy.deepcopy(data.meta_info),
    )


def make_worker_config():
    full = OmegaConf.load(CONFIG)
    resolved = OmegaConf.to_container(full.actor_rollout_ref, resolve=True)
    cfg = OmegaConf.create(resolved)
    cfg.model.path = MODEL
    cfg.model.use_remove_padding = True
    cfg.model.use_shm = False
    cfg.model.use_fused_kernels = False
    cfg.model.trust_remote_code = True
    cfg.actor.strategy = "fsdp"
    cfg.actor.ppo_mini_batch_size = 8
    cfg.actor.ppo_micro_batch_size = None
    cfg.actor.ppo_micro_batch_size_per_gpu = 1
    cfg.actor.ppo_max_token_len_per_gpu = 57344
    cfg.actor.ppo_epochs = 1
    cfg.actor.use_dynamic_bsz = False
    cfg.actor.ulysses_sequence_parallel_size = 2
    cfg.actor.grad_clip = 1.0
    cfg.actor.fsdp_config.param_offload = True
    cfg.actor.fsdp_config.optimizer_offload = True
    cfg.actor.optim.total_training_steps = 1
    cfg.rollout.n = 4
    cfg.rollout.temperature = 0.3
    cfg.rollout.log_prob_micro_batch_size = None
    cfg.rollout.log_prob_micro_batch_size_per_gpu = 1
    cfg.rollout.log_prob_max_token_len_per_gpu = 57344
    cfg.rollout.log_prob_use_dynamic_bsz = False
    return cfg


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    torch.cuda.set_device(rank)
    torch.cuda.reset_peak_memory_stats()

    cfg = make_worker_config()
    worker = ActorRolloutRefWorker(cfg, role="actor")
    coord = tuple(int(x) for x in worker.ulysses_device_mesh.get_coordinate())
    dp, sp = coord
    assert coord == (rank // 2, rank % 2)
    marker(rank, "worker-constructed", dp=dp, sp=sp)

    worker.init_model()
    active = flash_attention_integration._flash_attention_forward
    active_identity = f"{active.__module__}.{active.__name__}"
    sp_world = ulysses_patch_module.get_ulysses_sequence_parallel_world_size(
        worker.ulysses_device_mesh["sp"].get_group()
    )
    assert active_identity == (
        "verl.models.transformers.monkey_patch."
        "_ulysses_flash_attention_forward"
    )
    assert sp_world == 2
    marker(
        rank,
        "production-init-confirmed",
        active_attention=active_identity,
        sp_group_world=sp_world,
        dp=dp,
        sp=sp,
    )

    episodes, buckets, assignment, global_batch = build_fixture(
        worker.tokenizer.eos_token_id,
        worker.tokenizer.pad_token_id or worker.tokenizer.eos_token_id,
        effective_dp_degree=4,
        sp_size=2,
    )
    step0_indices = np.flatnonzero(
        global_batch.non_tensor_batch["optimizer_step_id"] == 0
    )
    step1_indices = np.flatnonzero(
        global_batch.non_tensor_batch["optimizer_step_id"] == 1
    )
    assert len(step0_indices) and len(step1_indices)
    step0_global = global_batch[step0_indices]
    production_temperature = float(worker.config.rollout.temperature)
    step0_global.meta_info["temperature"] = production_temperature
    assert set(int(x) for x in step0_global.non_tensor_batch["optimizer_step_id"]) == {0}
    assert len(step0_global) % world == 0
    canonical_local_batch = clone_dataproto(step0_global.chunk(world)[rank])
    assert np.all(canonical_local_batch.non_tensor_batch["target_dp_replica"] == dp)
    assert all("step1" not in str(x) for x in canonical_local_batch.non_tensor_batch["episode_id"])

    original_active_mask = (
        canonical_local_batch.batch["actor_token_weight"] > 0
    ).detach().clone()
    original_policy_mask = canonical_local_batch.batch["policy_mask"].detach().clone()
    response_width = original_policy_mask.shape[-1]
    original_attention_response = canonical_local_batch.batch["attention_mask"][
        :, -response_width:
    ].detach().clone()
    original_row_identity = {
        key: canonical_local_batch.non_tensor_batch[key].copy()
        for key in (
            "episode_id",
            "segment_id",
            "optimizer_step_id",
            "target_dp_replica",
            "is_dummy",
        )
    }
    original_physical_rows = len(canonical_local_batch)
    old_input = clone_dataproto(canonical_local_batch)
    actor_input = clone_dataproto(canonical_local_batch)

    real_mask_global = ~step0_global.non_tensor_batch["is_dummy"]
    dummy_mask_global = step0_global.non_tensor_batch["is_dummy"]
    global_real_weight = float(
        step0_global.batch["actor_token_weight"][real_mask_global].sum()
    )
    global_dummy_weight = float(
        step0_global.batch["actor_token_weight"][dummy_mask_global].sum()
    )
    assert abs(global_real_weight - 4.0) < 1e-5
    assert global_dummy_weight == 0.0

    # Episode scalar-advantage invariant across all real segments in step 0.
    advantage_by_episode = {}
    for row in range(len(step0_global)):
        if bool(step0_global.non_tensor_batch["is_dummy"][row]):
            continue
        episode_id = str(step0_global.non_tensor_batch["episode_id"][row])
        policy = step0_global.batch["policy_mask"][row].bool()
        values = torch.unique(step0_global.batch["advantages"][row][policy])
        assert values.numel() == 1
        value = float(values.item())
        if episode_id in advantage_by_episode:
            assert advantage_by_episode[episode_id] == value
        advantage_by_episode[episode_id] = value

    marker(
        rank,
        "old-logprob-start",
        local_rows=original_physical_rows,
        step0_global_rows=len(step0_global),
        excluded_step1_rows=len(step1_indices),
    )
    old_output = worker.compute_compression_log_prob_d2(old_input)
    old = old_output.batch["old_log_probs"]
    old_temperature = float(old_output.meta_info["temperature"])
    current_temperature = float(actor_input.meta_info["temperature"])
    assert old_temperature == current_temperature == production_temperature
    collector_restored_rows = old.shape[0]
    expanded_logical_rows = len(old_input)
    assert collector_restored_rows == original_physical_rows
    assert old.shape == original_active_mask.shape
    old_finite = finite_counts(old, original_active_mask)
    assert old_finite["nan"] == old_finite["posinf"] == old_finite["neginf"] == 0
    actor_input.batch["old_log_probs"] = old.to(actor_input.batch.device)
    marker(
        rank,
        "old-logprob-finite",
        original_physical_rows=original_physical_rows,
        expanded_logical_rows=expanded_logical_rows,
        collector_restored_rows=collector_restored_rows,
        **old_finite,
    )

    # The formal actor objective has KL disabled; do not instantiate a ref model.
    assert worker.config.actor.use_kl_loss is False

    counters = {
        "forward": 0,
        "backward": 0,
        "optimizer_step": 0,
        "zero_grad": 0,
    }
    current_reports = []
    ppo_reports = []
    trim_reports = []
    grad_report = {}
    optimizer_diagnostic = {}

    original_forward = worker.actor._forward_micro_batch
    def recording_forward(*args, **kwargs):
        entropy, log_prob = original_forward(*args, **kwargs)
        counters["forward"] += 1
        current_reports.append(
            {
                "temperature": float(kwargs["temperature"]),
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
                "requires_grad": bool(log_prob.requires_grad),
                "grad_fn": type(log_prob.grad_fn).__name__ if log_prob.grad_fn else None,
                "log_prob": finite_counts(log_prob),
                "entropy": finite_counts(entropy) if entropy is not None else None,
            }
        )
        return entropy, log_prob
    worker.actor._forward_micro_batch = recording_forward

    original_weighted_loss = dp_actor_module.compute_policy_loss_weighted
    def recording_weighted_loss(**kwargs):
        outputs = original_weighted_loss(**kwargs)
        pg_loss, clipfrac, ppo_kl, lower_clipfrac = outputs
        weights = kwargs["actor_token_weight"]
        active = weights > 0
        if bool(active.any()):
            log_ratio = kwargs["log_prob"] - kwargs["old_log_prob"]
            ratio = torch.exp(log_ratio)
            grad_lp = torch.autograd.grad(
                pg_loss,
                kwargs["log_prob"],
                retain_graph=True,
                allow_unused=True,
            )[0]
            ppo_reports.append(
                {
                    "active_tokens": int(active.sum()),
                    "weight_sum": float(weights.sum().detach().cpu()),
                    "ratio_min": float(ratio[active].min().detach().cpu()),
                    "ratio_max": float(ratio[active].max().detach().cpu()),
                    "ratio_mean": float(ratio[active].mean().detach().cpu()),
                    "max_abs_ratio_minus_one": float(
                        (ratio[active] - 1).abs().max().detach().cpu()
                    ),
                    "clipfrac": float(clipfrac.detach().cpu()),
                    "lower_clipfrac": float(lower_clipfrac.detach().cpu()),
                    "ppo_kl": float(ppo_kl.detach().cpu()),
                    "pg_loss": float(pg_loss.detach().cpu()),
                    "pg_loss_requires_grad": bool(pg_loss.requires_grad),
                    "pg_loss_grad_fn": type(pg_loss.grad_fn).__name__ if pg_loss.grad_fn else None,
                    "grad_loss_to_logprob": None if grad_lp is None else finite_counts(grad_lp),
                }
            )
        return outputs
    dp_actor_module.compute_policy_loss_weighted = recording_weighted_loss

    original_backward = torch.autograd.backward
    def recording_backward(*args, **kwargs):
        counters["backward"] += 1
        return original_backward(*args, **kwargs)
    torch.autograd.backward = recording_backward

    original_trim = dp_actor_module._trim_compression_microbatch
    def recording_trim(batch, prompt_width, response_width):
        trimmed, stats = original_trim(batch, prompt_width, response_width)
        trim_reports.append(
            {
                "p_carrier": int(prompt_width),
                "r_carrier": int(response_width),
                "p_local": int(stats["prompt_local_width"]),
                "r_local": int(stats["response_local_width"]),
                "total_nnz": int(stats["total_nnz"]),
                "sp_pad": int((-stats["total_nnz"]) % 2),
            }
        )
        return trimmed, stats
    dp_actor_module._trim_compression_microbatch = recording_trim

    optimizer = worker.actor_optimizer
    original_step = optimizer.step
    original_zero_grad = optimizer.zero_grad
    named_parameters = dict(worker.actor_module_fsdp.named_parameters())
    optional_name_by_id = {id(value): name for name, value in named_parameters.items()}

    def optimizer_state_report(value):
        state = optimizer.state.get(value, {})
        result = {}
        for key in ("step", "exp_avg", "exp_avg_sq"):
            item = state.get(key)
            if item is None:
                result[key] = None
            elif torch.is_tensor(item):
                cpu_item = item.detach().float().cpu()
                result[key] = (
                    float(cpu_item.item()) if cpu_item.numel() == 1
                    else {"norm": float(cpu_item.norm()), "finite": bool(torch.isfinite(cpu_item).all())}
                )
            else:
                result[key] = float(item) if isinstance(item, (int, float)) else str(item)
        return result

    def recording_step(*args, **kwargs):
        counters["optimizer_step"] += 1
        parameter_stats = {
            "optimizer_param_tensors": 0,
            "requires_grad_tensors": 0,
            "grad_none_tensors": 0,
            "finite_grad_tensors": 0,
            "zero_grad_tensors": 0,
            "nonzero_grad_tensors": 0,
            "nonzero_grad_elements": 0,
            "nonfinite_grad_tensors": 0,
        }
        selected = None
        selected_name = None
        selected_group_idx = None
        selected_param_idx = None
        for group_idx, group in enumerate(optimizer.param_groups):
            for param_idx, value in enumerate(group["params"]):
                parameter_stats["optimizer_param_tensors"] += 1
                if value.requires_grad:
                    parameter_stats["requires_grad_tensors"] += 1
                grad = value.grad
                if grad is None:
                    parameter_stats["grad_none_tensors"] += 1
                    continue
                if not bool(torch.isfinite(grad).all()):
                    parameter_stats["nonfinite_grad_tensors"] += 1
                    continue
                parameter_stats["finite_grad_tensors"] += 1
                nonzero = grad != 0
                nonzero_count = int(nonzero.sum())
                if nonzero_count == 0:
                    parameter_stats["zero_grad_tensors"] += 1
                    continue
                parameter_stats["nonzero_grad_tensors"] += 1
                parameter_stats["nonzero_grad_elements"] += nonzero_count
                if selected is None:
                    selected = value
                    selected_group_idx = group_idx
                    selected_param_idx = param_idx
                    selected_name = optional_name_by_id.get(id(value))

        param_groups = [
            {
                "index": index,
                "lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "betas": [float(x) for x in group.get("betas", ())],
                "eps": float(group.get("eps", 0.0)),
                "parameter_count": len(group["params"]),
            }
            for index, group in enumerate(optimizer.param_groups)
        ]
        scheduler = worker.actor_lr_scheduler
        configured_warmup_steps = int(getattr(worker.config.actor.optim, "lr_warmup_steps", -1))
        effective_warmup_steps = configured_warmup_steps
        if effective_warmup_steps < 0:
            effective_warmup_steps = int(
                float(getattr(worker.config.actor.optim, "lr_warmup_steps_ratio", 0.0))
                * int(getattr(worker.config.actor.optim, "total_training_steps", 0))
            )
        scheduler_report = {
            "last_epoch": int(scheduler.last_epoch),
            "step_count": int(getattr(scheduler, "_step_count", -1)),
            "configured_warmup_steps": configured_warmup_steps,
            "effective_warmup_steps": effective_warmup_steps,
        }
        optimizer_diagnostic.update(
            {
                "param_groups": param_groups,
                "scheduler_before": scheduler_report,
                "parameter_stats": parameter_stats,
                "fsdp": {
                    "use_orig_params": bool(getattr(worker.actor_module_fsdp, "_use_orig_params", False)),
                    "sharding_strategy": str(getattr(worker.actor_module_fsdp, "sharding_strategy", None)),
                    "mixed_precision_param_dtype": str(
                        getattr(
                            getattr(
                                worker.actor_module_fsdp,
                                "mixed_precision",
                                getattr(worker.actor_module_fsdp, "_mixed_precision", None),
                            ),
                            "param_dtype",
                            None,
                        )
                    ),
                    "named_parameter_count": len(named_parameters),
                    "optimizer_parameter_count": sum(
                        len(group["params"]) for group in optimizer.param_groups
                    ),
                },
            }
        )
        assert parameter_stats["nonfinite_grad_tensors"] == 0
        marker(
            rank,
            "optimizer-owned-gradient-diagnostic",
            param_groups=param_groups,
            scheduler=scheduler_report,
            parameter_stats=parameter_stats,
            fsdp=optimizer_diagnostic["fsdp"],
        )
        assert selected is not None

        flat_grad = selected.grad.detach().reshape(-1)
        selected_index = int(torch.argmax(flat_grad.abs()).item())
        before = selected.detach().float().cpu().clone()
        before_flat = before.reshape(-1)
        before_value = float(before_flat[selected_index])
        grad_value = float(flat_grad[selected_index].detach().float().cpu())
        state_before = optimizer_state_report(selected)
        result = original_step(*args, **kwargs)

        after = selected.detach().float().cpu().clone()
        after_flat = after.reshape(-1)
        delta = after - before
        delta_flat = delta.reshape(-1)
        fsdp_module = worker.actor_module_fsdp
        mixed_precision = getattr(
            fsdp_module, "mixed_precision", getattr(fsdp_module, "_mixed_precision", None)
        )
        optimizer_diagnostic["selected_parameter"] = {
            "name": selected_name,
            "group_idx": selected_group_idx,
            "param_idx": selected_param_idx,
            "shape": list(selected.shape),
            "dtype": str(selected.dtype),
            "device": str(selected.device),
            "type": f"{type(selected).__module__}.{type(selected).__name__}",
            "is_flat_parameter": type(selected).__name__ == "FlatParameter",
            "use_orig_params": bool(getattr(fsdp_module, "_use_orig_params", False)),
            "mixed_precision_param_dtype": str(getattr(mixed_precision, "param_dtype", None)),
            "optimizer_owns_parameter": True,
            "grad_max_abs": float(flat_grad.abs().max().detach().float().cpu()),
            "grad_norm": float(flat_grad.float().norm().detach().cpu()),
            "selected_index": selected_index,
            "grad_at_index": grad_value,
            "before_at_index": before_value,
            "after_at_index": float(after_flat[selected_index]),
            "delta_at_index": float(delta_flat[selected_index]),
            "parameter_before_norm": float(before.norm()),
            "parameter_after_norm": float(after.norm()),
            "parameter_magnitude_max": float(before.abs().max()),
            "changed_elements": int((delta != 0).sum()),
            "max_abs_delta": float(delta.abs().max()),
            "l1_delta": float(delta.abs().sum()),
            "l2_delta": float(delta.norm()),
            "state_before": state_before,
            "state_after": optimizer_state_report(selected),
        }
        return result
    def recording_zero_grad(*args, **kwargs):
        counters["zero_grad"] += 1
        return original_zero_grad(*args, **kwargs)
    optimizer.step = recording_step
    optimizer.zero_grad = recording_zero_grad

    fsdp_module = worker.actor_module_fsdp
    original_clip = fsdp_module.clip_grad_norm_
    def recording_clip(max_norm, *args, **kwargs):
        elements = nan = posinf = neginf = 0
        for value in fsdp_module.parameters():
            if value.grad is None:
                continue
            grad = value.grad.detach()
            elements += grad.numel()
            nan += int(torch.isnan(grad).sum())
            posinf += int(torch.isposinf(grad).sum())
            neginf += int(torch.isneginf(grad).sum())
        norm = original_clip(max_norm, *args, **kwargs)
        grad_report.update(
            {
                "elements": elements,
                "nan": nan,
                "posinf": posinf,
                "neginf": neginf,
                "pre_clip_norm": float(norm.detach().cpu()),
                "grad_clip": float(max_norm),
                "clip_triggered": bool(float(norm.detach().cpu()) > float(max_norm)),
            }
        )
        return norm
    fsdp_module.clip_grad_norm_ = recording_clip

    torch.cuda.reset_peak_memory_stats()
    marker(rank, "update-start")
    update_started = time.perf_counter()
    try:
        update_output = worker.update_compression_actor_d2(actor_input)
    finally:
        torch.autograd.backward = original_backward
        dp_actor_module._trim_compression_microbatch = original_trim
        dp_actor_module.compute_policy_loss_weighted = original_weighted_loss
    update_seconds = time.perf_counter() - update_started
    marker(rank, "update-done", seconds=update_seconds)

    metrics = update_output.meta_info["metrics"]
    assert counters["optimizer_step"] == 1
    assert counters["forward"] == counters["backward"]
    assert grad_report["nan"] == grad_report["posinf"] == grad_report["neginf"] == 0
    assert np.isfinite(grad_report["pre_clip_norm"])
    for item in current_reports:
        assert item["log_prob"]["nan"] == 0
        assert item["log_prob"]["posinf"] == 0
        assert item["log_prob"]["neginf"] == 0

    memory = {
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
    }
    local_report = {
        "rank": rank,
        "dp": dp,
        "sp": sp,
        "active_attention": active_identity,
        "sp_group_world": sp_world,
        "old": old_finite,
        "temperature": {
            "production": production_temperature,
            "old": old_temperature,
            "current": current_temperature,
        },
        "global_real_weight": global_real_weight,
        "global_dummy_weight": global_dummy_weight,
        "advantage_by_episode": advantage_by_episode,
        "ref_skipped_use_kl_loss_false": True,
        "current": current_reports,
        "ppo": ppo_reports,
        "counters": counters,
        "grad": grad_report,
        "optimizer_diagnostic": optimizer_diagnostic,
        "trim": trim_reports,
        "metrics": metrics,
        "memory": memory,
        "update_seconds": update_seconds,
        "original_physical_rows": original_physical_rows,
        "old_expanded_logical_rows": expanded_logical_rows,
        "collector_restored_rows": collector_restored_rows,
        "original_policy_tokens": int(original_policy_mask.sum()),
        "original_attention_response_tokens": int(original_attention_response.sum()),
        "original_row_identity": {
            key: value.tolist() for key, value in original_row_identity.items()
        },
        "local_episode_ids": [str(x) for x in original_row_identity["episode_id"]],
        "local_optimizer_step_ids": [int(x) for x in original_row_identity["optimizer_step_id"]],
    }
    reports = [None] * world
    dist.all_gather_object(reports, local_report)
    if rank == 0:
        # Persist the full Stage 8E summary before any final diagnostic assertion.
        print(json.dumps({"stage8e": "diagnostics", "reports": reports}, indent=2), flush=True)
        assert all(item["counters"]["optimizer_step"] == 1 for item in reports)
        assert len({item["counters"]["forward"] for item in reports}) == 1
        assert len({item["counters"]["backward"] for item in reports}) == 1
        assert all(set(item["local_optimizer_step_ids"]) == {0} for item in reports)
        assert all(item["temperature"]["old"] == item["temperature"]["current"] for item in reports)
        assert sum(
            report["grad_loss_to_logprob"]["nonzero"]
            for item in reports
            for report in item["ppo"]
            if report["grad_loss_to_logprob"] is not None
        ) > 0
        assert all(
            all(group["lr"] > 0 for group in item["optimizer_diagnostic"]["param_groups"])
            for item in reports
        )
        for item in reports:
            selected_report = item["optimizer_diagnostic"]["selected_parameter"]
            before_step = selected_report["state_before"]["step"]
            after_step = selected_report["state_after"]["step"]
            before_step = 0.0 if before_step is None else float(before_step)
            assert after_step is not None and float(after_step) > before_step
        assert all(
            item["optimizer_diagnostic"]["selected_parameter"]["max_abs_delta"] > 0
            for item in reports
        )
        print(json.dumps({"stage8e": "passed", "reports": reports}, indent=2), flush=True)
    dist.barrier()
    marker(rank, "stage8e-complete")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
