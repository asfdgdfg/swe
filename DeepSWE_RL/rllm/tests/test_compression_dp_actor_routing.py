import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.workers.actor import dp_actor as dp_actor_module
from verl.workers.actor.dp_actor import DataParallelPPOActor


class _ScalarActorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))


def _config(*, micro_batch_size, dynamic=False, entropy_coeff=0.0, use_kl=False):
    return OmegaConf.create(
        {
            "use_kl_loss": use_kl,
            "ppo_mini_batch_size": 4,
            "ppo_micro_batch_size_per_gpu": micro_batch_size,
            "use_dynamic_bsz": dynamic,
            "ppo_max_token_len_per_gpu": 1024,
            "ppo_epochs": 1,
            "clip_ratio": 0.2,
            "clip_ratio_low": None,
            "clip_ratio_high": None,
            "clip_ratio_c": 3.0,
            "entropy_coeff": entropy_coeff,
            "loss_agg_mode": "token-mean",
            "kl_loss_type": "k1",
            "kl_loss_coef": 0.1,
        }
    )


def _batch(*, compression=True, include_weights=True):
    tensors = {
        "responses": torch.ones(4, 2, dtype=torch.long),
        "input_ids": torch.ones(4, 4, dtype=torch.long),
        "attention_mask": torch.ones(4, 4, dtype=torch.long),
        "position_ids": torch.arange(4).repeat(4, 1),
        "old_log_probs": torch.zeros(4, 2),
        "advantages": torch.ones(4, 2),
        "ref_log_prob": torch.full((4, 2), -1.0),
    }
    if include_weights:
        tensors["policy_mask"] = torch.ones(4, 2, dtype=torch.long)
        tensors["actor_token_weight"] = torch.full((4, 2), 1 / 8)
    non_tensors = {}
    if compression:
        non_tensors["optimizer_step_id"] = np.zeros(4, dtype=np.int64)
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info={
            "temperature": 1.0,
            "compression_segment_training": compression,
        },
    )


def _run_actor(
    monkeypatch,
    *,
    micro_batch_size,
    dynamic=False,
    partitions=None,
    compression=True,
    entropy_coeff=0.0,
    use_kl=False,
):
    actor_module = _ScalarActorModule()
    optimizer = torch.optim.SGD(actor_module.parameters(), lr=0.1)
    actor = object.__new__(DataParallelPPOActor)
    actor.config = _config(
        micro_batch_size=micro_batch_size,
        dynamic=dynamic,
        entropy_coeff=entropy_coeff,
        use_kl=use_kl,
    )
    actor.actor_module = actor_module
    actor.actor_optimizer = optimizer
    actor.ulysses_sequence_parallel_size = 1
    captured = {}

    def forward(micro_batch, temperature, calculate_entropy):
        shape = micro_batch["old_log_probs"].shape
        entropy = torch.full(shape, 2.0) if calculate_entropy else None
        return entropy, actor_module.value.expand(shape)

    def optimizer_step():
        captured["grad"] = actor_module.value.grad.detach().clone()
        return torch.tensor(0.0)

    actor._forward_micro_batch = forward
    actor._optimizer_step = optimizer_step
    monkeypatch.setattr(
        dp_actor_module,
        "get_torch_device",
        lambda: SimpleNamespace(current_device=lambda: "cpu"),
    )
    if dynamic:
        def rearrange(batch, max_token_len):
            chunks = [batch[start:end] for start, end in partitions]
            indices = [list(range(start, end)) for start, end in partitions]
            return chunks, indices

        monkeypatch.setattr(dp_actor_module, "rearrange_micro_batches", rearrange)

    update_policy = inspect.unwrap(DataParallelPPOActor.update_policy)
    metrics = update_policy(actor, _batch(compression=compression, include_weights=compression))
    return captured["grad"], metrics


def test_fixed_microbatch_partition_does_not_rescale_weighted_gradient(monkeypatch):
    one_micro_grad, one_metrics = _run_actor(monkeypatch, micro_batch_size=4)
    four_micro_grad, four_metrics = _run_actor(monkeypatch, micro_batch_size=1)
    assert one_micro_grad.item() == pytest.approx(-1.0)
    assert four_micro_grad.item() == pytest.approx(-1.0)
    assert one_micro_grad.item() == pytest.approx(four_micro_grad.item())
    assert one_metrics["actor/pg_loss"] == pytest.approx([-1.0])
    assert four_metrics["actor/pg_loss"] == pytest.approx([-1.0])


def test_compression_dynamic_batch_fails_before_forward_or_optimizer(monkeypatch):
    actor_module = _ScalarActorModule()
    optimizer = torch.optim.SGD(actor_module.parameters(), lr=0.1)
    actor = object.__new__(DataParallelPPOActor)
    actor.config = _config(micro_batch_size=1, dynamic=True)
    actor.actor_module = actor_module
    actor.actor_optimizer = optimizer
    actor.ulysses_sequence_parallel_size = 1
    calls = {"forward": 0, "zero_grad": 0, "step": 0}

    def forward(*args, **kwargs):
        calls["forward"] += 1
        raise AssertionError("forward must not run")

    def zero_grad(*args, **kwargs):
        calls["zero_grad"] += 1

    def step(*args, **kwargs):
        calls["step"] += 1

    actor._forward_micro_batch = forward
    monkeypatch.setattr(optimizer, "zero_grad", zero_grad)
    monkeypatch.setattr(optimizer, "step", step)
    update_policy = inspect.unwrap(DataParallelPPOActor.update_policy)
    with pytest.raises(RuntimeError, match="requires use_dynamic_bsz=False"):
        update_policy(actor, _batch(compression=True, include_weights=True))
    assert calls == {"forward": 0, "zero_grad": 0, "step": 0}


def test_feature_disabled_dynamic_baseline_still_runs(monkeypatch):
    grad, metrics = _run_actor(
        monkeypatch,
        micro_batch_size=1,
        dynamic=True,
        partitions=[(0, 1), (1, 4)],
        compression=False,
    )
    assert grad.item() == pytest.approx(-1.0)
    # Baseline dynamic actor metrics record the unscaled per-microbatch
    # policy loss. Row-count scaling is applied only to the backward loss.
    assert metrics["actor/pg_loss"] == pytest.approx([-1.0, -1.0])


def test_weighted_entropy_kl_and_metrics_sum_once_per_optimizer_step(monkeypatch):
    grad, metrics = _run_actor(
        monkeypatch,
        micro_batch_size=1,
        entropy_coeff=0.1,
        use_kl=True,
    )
    assert grad.item() == pytest.approx(-0.9)
    assert metrics["actor/pg_loss"] == pytest.approx([-1.0])
    assert metrics["actor/entropy_loss"] == pytest.approx([2.0])
    assert metrics["actor/kl_loss"] == pytest.approx([1.0])
    assert metrics["actor/kl_coef"] == pytest.approx([0.1])


def test_compression_flag_requires_weighted_fields_before_forward():
    actor_module = _ScalarActorModule()
    actor = object.__new__(DataParallelPPOActor)
    actor.config = _config(micro_batch_size=4)
    actor.actor_module = actor_module
    actor.actor_optimizer = torch.optim.SGD(actor_module.parameters(), lr=0.1)
    update_policy = inspect.unwrap(DataParallelPPOActor.update_policy)
    with pytest.raises(
        ValueError,
        match="missing required fields:.*actor_token_weight.*policy_mask",
    ):
        update_policy(actor, _batch(compression=True, include_weights=False))


def test_feature_disabled_routes_to_unchanged_baseline_helper_and_scaling(monkeypatch):
    calls = {"baseline": 0, "weighted": 0}
    original_baseline = dp_actor_module.compute_policy_loss
    original_weighted = dp_actor_module.compute_policy_loss_weighted

    def baseline(*args, **kwargs):
        calls["baseline"] += 1
        return original_baseline(*args, **kwargs)

    def weighted(*args, **kwargs):
        calls["weighted"] += 1
        return original_weighted(*args, **kwargs)

    monkeypatch.setattr(dp_actor_module, "compute_policy_loss", baseline)
    monkeypatch.setattr(dp_actor_module, "compute_policy_loss_weighted", weighted)
    grad, metrics = _run_actor(
        monkeypatch, micro_batch_size=1, compression=False
    )
    assert calls == {"baseline": 4, "weighted": 0}
    assert grad.item() == pytest.approx(-1.0)
    assert metrics["actor/pg_loss"] == pytest.approx([-1.0] * 4)
