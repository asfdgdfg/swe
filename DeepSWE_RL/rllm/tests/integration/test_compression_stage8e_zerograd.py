"""Single-GPU real-Qwen autograd diagnosis for Stage 8E zero gradients."""

import copy
import json
import os

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from tests.integration.test_compression_d2_logprob_smoke import build_fixture
import verl.workers.actor.dp_actor as dp_actor_module
from verl.workers.fsdp_workers import ActorRolloutRefWorker


ROOT = "/home/yyk/yyk11/zhongtianyang/memory/SWE-Master/DeepSWE_RL/rllm"
MODEL = "/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B"
CONFIG = f"{ROOT}/rllm/trainer/config/ppo_trainer.yaml"


def clone_dataproto(data):
    return type(data)(
        batch=data.batch.clone(),
        non_tensor_batch={key: value.copy() for key, value in data.non_tensor_batch.items()},
        meta_info=copy.deepcopy(data.meta_info),
    )


def finite_report(tensor):
    return {
        "count": int(tensor.numel()),
        "finite": bool(torch.isfinite(tensor).all()),
        "nonzero": int((tensor != 0).sum()),
        "max_abs": float(tensor.detach().abs().max().float().cpu()) if tensor.numel() else 0.0,
    }


def config():
    full = OmegaConf.load(CONFIG)
    cfg = OmegaConf.create(OmegaConf.to_container(full.actor_rollout_ref, resolve=True))
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
    cfg.actor.ulysses_sequence_parallel_size = 1
    cfg.actor.grad_clip = 1.0
    cfg.actor.fsdp_config.param_offload = True
    cfg.actor.fsdp_config.optimizer_offload = True
    cfg.actor.optim.total_training_steps = 1
    cfg.rollout.n = 1
    cfg.rollout.temperature = 0.3
    cfg.rollout.log_prob_micro_batch_size = None
    cfg.rollout.log_prob_micro_batch_size_per_gpu = 1
    cfg.rollout.log_prob_max_token_len_per_gpu = 57344
    cfg.rollout.log_prob_use_dynamic_bsz = False
    return cfg


class DiagnosticComplete(RuntimeError):
    pass


def main():
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 1
    torch.cuda.set_device(0)
    worker = ActorRolloutRefWorker(config(), role="actor")
    worker.init_model()

    episodes, buckets, assignment, global_batch = build_fixture(
        worker.tokenizer.eos_token_id,
        worker.tokenizer.pad_token_id or worker.tokenizer.eos_token_id,
        effective_dp_degree=1,
        sp_size=1,
    )
    indices = np.flatnonzero(global_batch.non_tensor_batch["optimizer_step_id"] == 0)
    canonical = clone_dataproto(global_batch[indices])

    episode_report = []
    for episode in buckets[0]:
        rows = np.flatnonzero(canonical.non_tensor_batch["episode_id"] == episode.episode_id)
        policy_tokens = 0
        weight_sum = 0.0
        segment_reports = []
        for row in rows:
            policy = canonical.batch["policy_mask"][row].bool()
            advantages = canonical.batch["advantages"][row][policy]
            weights = canonical.batch["actor_token_weight"][row]
            policy_tokens += int(policy.sum())
            weight_sum += float(weights.sum())
            segment_reports.append(
                {
                    "segment_id": int(canonical.non_tensor_batch["segment_id"][row]),
                    "policy_tokens": int(policy.sum()),
                    "weight_sum": float(weights.sum()),
                    "advantage_unique": torch.unique(advantages).tolist(),
                }
            )
        episode_report.append(
            {
                "episode_id": episode.episode_id,
                "group_uid": episode.group_uid,
                "scalar_advantage": episode.scalar_advantage,
                "policy_token_count": policy_tokens,
                "segment_count": len(rows),
                "weight_sum": weight_sum,
                "segments": segment_reports,
            }
        )

    old_input = clone_dataproto(canonical)
    old_output = worker.compute_compression_log_prob_d1(old_input)
    old = old_output.batch["old_log_probs"]
    original_active = canonical.batch["actor_token_weight"] > 0
    assert old.shape == original_active.shape
    old_active = old[original_active.to(old.device)]
    assert torch.isfinite(old_active).all()

    optimizer_params = [p for g in worker.actor_optimizer.param_groups for p in g["params"]]
    lm_head = dict(worker.actor_module_fsdp.named_modules()).get("_fsdp_wrapped_module.lm_head")
    if lm_head is None:
        lm_head = next(module for name, module in worker.actor_module_fsdp.named_modules() if name.endswith("lm_head"))
    state = {"latest_logits": None, "forward": [], "math": None, "single_token": None, "weighted_params": None}

    def logits_hook(module, inputs, output):
        state["latest_logits"] = output

    hook_handle = lm_head.register_forward_hook(logits_hook)
    original_forward = worker.actor._forward_micro_batch
    def forward_wrapper(*args, **kwargs):
        grad_enabled = torch.is_grad_enabled()
        inference = torch.is_inference_mode_enabled()
        entropy, log_prob = original_forward(*args, **kwargs)
        state["forward"].append(
            {
                "grad_enabled": grad_enabled,
                "inference_mode": inference,
                "log_prob_requires_grad": bool(log_prob.requires_grad),
                "log_prob_grad_fn": type(log_prob.grad_fn).__name__ if log_prob.grad_fn else None,
                "entropy_requires_grad": bool(entropy.requires_grad) if entropy is not None else None,
            }
        )
        return entropy, log_prob
    worker.actor._forward_micro_batch = forward_wrapper

    original_weighted = dp_actor_module.compute_policy_loss_weighted
    def diagnostic_weighted(**kwargs):
        outputs = original_weighted(**kwargs)
        pg_loss, clipfrac, ppo_kl, lower_clipfrac = outputs
        log_prob = kwargs["log_prob"]
        old_log_prob = kwargs["old_log_prob"]
        advantages = kwargs["advantages"]
        weights = kwargs["actor_token_weight"]
        active = weights > 0
        if not bool(active.any()):
            return outputs
        log_ratio = log_prob - old_log_prob
        ratio = torch.exp(log_ratio)
        clip_low = kwargs["cliprange_low"]
        clip_high = kwargs["cliprange_high"]
        unclipped = -advantages * ratio
        clipped = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
        first_stage = torch.maximum(unclipped, clipped)
        dual = -advantages * kwargs["clip_ratio_c"]
        chosen = torch.where(advantages < 0, torch.min(dual, first_stage), first_stage)
        grad_lp = torch.autograd.grad(pg_loss, log_prob, retain_graph=True, allow_unused=True)[0]
        latest_logits = state["latest_logits"]
        grad_logits = torch.autograd.grad(pg_loss, latest_logits, retain_graph=True, allow_unused=True)[0]

        active_flat = active.reshape(-1).nonzero(as_tuple=False)[0, 0]
        single_loss = -log_prob.reshape(-1)[active_flat]
        single_grads = torch.autograd.grad(
            single_loss, optimizer_params, retain_graph=True, allow_unused=True
        )
        weighted_grads = torch.autograd.grad(
            pg_loss, optimizer_params, retain_graph=True, allow_unused=True
        )
        def parameter_grad_report(grads):
            visible = [g for g in grads if g is not None]
            return {
                "grad_none": len(grads) - len(visible),
                "grad_tensors": len(visible),
                "nonzero_tensors": sum(int(bool((g != 0).any())) for g in visible),
                "nonzero_elements": sum(int((g != 0).sum()) for g in visible),
                "finite": all(bool(torch.isfinite(g).all()) for g in visible),
                "max_abs": max((float(g.abs().max().float().cpu()) for g in visible), default=0.0),
            }
        positive = active & (advantages > 0)
        negative = active & (advantages < 0)
        state["math"] = {
            "update_policy_grad_enabled": torch.is_grad_enabled(),
            "update_policy_inference_mode": torch.is_inference_mode_enabled(),
            "old_requires_grad": bool(old_log_prob.requires_grad),
            "old_grad_fn": type(old_log_prob.grad_fn).__name__ if old_log_prob.grad_fn else None,
            "current_requires_grad": bool(log_prob.requires_grad),
            "current_grad_fn": type(log_prob.grad_fn).__name__ if log_prob.grad_fn else None,
            "loss": float(pg_loss.detach().cpu()),
            "loss_requires_grad": bool(pg_loss.requires_grad),
            "loss_grad_fn": type(pg_loss.grad_fn).__name__ if pg_loss.grad_fn else None,
            "weight_sum": float(weights.sum().detach().cpu()),
            "weighted_numerator": float((chosen * weights).sum().detach().cpu()),
            "active_tokens": int(active.sum()),
            "positive_active_tokens": int(positive.sum()),
            "negative_active_tokens": int(negative.sum()),
            "positive_weighted_contribution": float((chosen * weights * positive).sum().detach().cpu()),
            "negative_weighted_contribution": float((chosen * weights * negative).sum().detach().cpu()),
            "ratio_min": float(ratio[active].min().detach().cpu()),
            "ratio_max": float(ratio[active].max().detach().cpu()),
            "ratio_mean": float(ratio[active].mean().detach().cpu()),
            "ratio_equal_one": int((ratio[active] == 1).sum()),
            "ratio_not_equal_one": int((ratio[active] != 1).sum()),
            "clipfrac": float(clipfrac.detach().cpu()),
            "lower_clipfrac": float(lower_clipfrac.detach().cpu()),
            "ppo_kl": float(ppo_kl.detach().cpu()),
            "unclipped_active": finite_report(unclipped[active]),
            "clipped_active": finite_report(clipped[active]),
            "chosen_active": finite_report(chosen[active]),
            "grad_loss_to_logprob": None if grad_lp is None else finite_report(grad_lp),
            "grad_loss_to_logits": None if grad_logits is None else finite_report(grad_logits),
        }
        state["single_token"] = parameter_grad_report(single_grads)
        state["weighted_params"] = parameter_grad_report(weighted_grads)
        raise DiagnosticComplete

    dp_actor_module.compute_policy_loss_weighted = diagnostic_weighted
    pass_a = clone_dataproto(canonical)
    pass_a.batch["old_log_probs"] = old.to(pass_a.batch.device)
    try:
        worker.update_compression_actor_d1(pass_a)
    except DiagnosticComplete:
        pass
    finally:
        dp_actor_module.compute_policy_loss_weighted = original_weighted
        worker.actor_optimizer.zero_grad()

    # Fresh graph: run the unmodified production loss/backward, but replace the
    # optimizer call by a gradient inspection no-op.
    accumulated = {}
    original_step = worker.actor_optimizer.step
    def no_op_step(*args, **kwargs):
        grads = [p.grad for p in optimizer_params]
        visible = [g for g in grads if g is not None]
        accumulated.update(
            {
                "optimizer_step_executed": False,
                "grad_none": len(grads) - len(visible),
                "grad_tensors": len(visible),
                "zero_grad_tensors": sum(int(not bool((g != 0).any())) for g in visible),
                "nonzero_grad_tensors": sum(int(bool((g != 0).any())) for g in visible),
                "nonzero_grad_elements": sum(int((g != 0).sum()) for g in visible),
                "finite": all(bool(torch.isfinite(g).all()) for g in visible),
                "max_abs": max((float(g.abs().max().float().cpu()) for g in visible), default=0.0),
            }
        )
        return None
    worker.actor_optimizer.step = no_op_step
    pass_b = clone_dataproto(canonical)
    pass_b.batch["old_log_probs"] = old.to(pass_b.batch.device)
    update_output = worker.update_compression_actor_d1(pass_b)
    worker.actor_optimizer.step = original_step
    hook_handle.remove()

    advantages = [float(item["scalar_advantage"]) for item in episode_report]
    report = {
        "mode": "world1-D1-SP1-production-current-actor",
        "advantage_source": "integration fixture artificial scalar_advantage",
        "episodes": episode_report,
        "advantage_stats": {
            "min": min(advantages), "max": max(advantages),
            "mean": sum(advantages) / len(advantages),
            "nonzero_episodes": sum(value != 0 for value in advantages),
            "nonzero_policy_tokens": int(((canonical.batch["advantages"] != 0) & (canonical.batch["policy_mask"] == 1)).sum()),
        },
        "global_actor_token_weight_sum": float(canonical.batch["actor_token_weight"].sum()),
        "old_active": finite_report(old_active),
        "forward": state["forward"],
        "math": state["math"],
        "single_token_parameter_grad": state["single_token"],
        "weighted_loss_parameter_grad": state["weighted_params"],
        "production_accumulated_parameter_grad": accumulated,
        "metrics": update_output.meta_info["metrics"],
        "entropy_coeff": float(worker.config.actor.entropy_coeff),
        "use_kl_loss": bool(worker.config.actor.use_kl_loss),
    }
    print(json.dumps(report, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
