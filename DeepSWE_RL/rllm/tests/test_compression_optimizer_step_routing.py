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


def _config(*, ppo_epochs, ppo_mini_batch_size, micro_batch_size):
    return OmegaConf.create(
        {
            "use_kl_loss": False,
            "ppo_mini_batch_size": ppo_mini_batch_size,
            "ppo_micro_batch_size_per_gpu": micro_batch_size,
            "use_dynamic_bsz": False,
            "ppo_max_token_len_per_gpu": 1024,
            "ppo_epochs": ppo_epochs,
            "clip_ratio": 0.2,
            "clip_ratio_low": None,
            "clip_ratio_high": None,
            "clip_ratio_c": 3.0,
            "entropy_coeff": 0.0,
            "loss_agg_mode": "token-mean",
            "kl_loss_type": "k1",
            "kl_loss_coef": 0.0,
            "grad_clip": 1.0,
        }
    )


def _batch(step_ids):
    step_ids = np.asarray(step_ids, dtype=np.int64)
    rows = len(step_ids)
    row_ids = torch.arange(rows, dtype=torch.long)
    responses = torch.stack([torch.from_numpy(step_ids), row_ids], dim=-1)
    weights = torch.zeros(rows, 2)
    for step_id in np.unique(step_ids):
        step_rows = np.where(step_ids == step_id)[0]
        weights[step_rows] = 1.0 / (len(step_rows) * 2)
    return DataProto.from_dict(
        tensors={
            "responses": responses,
            "input_ids": torch.cat([torch.ones(rows, 2, dtype=torch.long), responses], dim=-1),
            "attention_mask": torch.ones(rows, 4, dtype=torch.long),
            "position_ids": torch.arange(4).repeat(rows, 1),
            "old_log_probs": torch.zeros(rows, 2),
            "advantages": torch.ones(rows, 2),
            "policy_mask": torch.ones(rows, 2, dtype=torch.long),
            "actor_token_weight": weights,
        },
        non_tensors={"optimizer_step_id": step_ids},
        meta_info={"temperature": 1.0, "compression_segment_training": True},
    )


def _run(monkeypatch, *, step_ids, ppo_epochs, ppo_mini_batch_size, micro_batch_size):
    actor_module = _ScalarActorModule()
    optimizer = torch.optim.SGD(actor_module.parameters(), lr=0.0)
    actor = object.__new__(DataParallelPPOActor)
    actor.config = _config(
        ppo_epochs=ppo_epochs,
        ppo_mini_batch_size=ppo_mini_batch_size,
        micro_batch_size=micro_batch_size,
    )
    actor.actor_module = actor_module
    actor.actor_optimizer = optimizer
    actor.ulysses_sequence_parallel_size = 1

    zero_grad_count = 0
    optimizer_step_count = 0
    pending_microbatches = []
    optimizer_steps = []
    events = []
    original_zero_grad = optimizer.zero_grad
    original_step = optimizer.step

    def zero_grad(*args, **kwargs):
        nonlocal zero_grad_count
        zero_grad_count += 1
        events.append("zero_grad")
        return original_zero_grad(*args, **kwargs)

    def step(*args, **kwargs):
        nonlocal optimizer_step_count, pending_microbatches
        optimizer_step_count += 1
        optimizer_steps.append(pending_microbatches)
        events.append("optimizer_step")
        pending_microbatches = []
        return original_step(*args, **kwargs)

    def forward(micro_batch, temperature, calculate_entropy):
        del temperature, calculate_entropy
        response = micro_batch["responses"]
        events.append(("forward", tuple(sorted(set(response[:, 0].tolist())))))
        pending_microbatches.append(
            {
                "step_ids": tuple(sorted(set(response[:, 0].tolist()))),
                "row_indices": tuple(response[:, 1].tolist()),
            }
        )
        shape = micro_batch["old_log_probs"].shape
        return None, actor_module.value.expand(shape)

    monkeypatch.setattr(optimizer, "zero_grad", zero_grad)
    monkeypatch.setattr(optimizer, "step", step)
    actor._forward_micro_batch = forward
    monkeypatch.setattr(
        dp_actor_module,
        "get_torch_device",
        lambda: SimpleNamespace(current_device=lambda: "cpu"),
    )
    update_policy = inspect.unwrap(DataParallelPPOActor.update_policy)
    update_policy(actor, _batch(step_ids))
    return {
        "zero_grad_count": zero_grad_count,
        "optimizer_step_count": optimizer_step_count,
        "optimizer_steps": optimizer_steps,
        "events": events,
    }


def _step_order(result):
    return [microbatches[0]["step_ids"][0] for microbatches in result["optimizer_steps"]]


def _rows_per_step(result):
    return [
        tuple(row for microbatch in microbatches for row in microbatch["row_indices"])
        for microbatches in result["optimizer_steps"]
    ]


def test_epoch_outer_step_inner_real_call_order(monkeypatch):
    result = _run(
        monkeypatch,
        step_ids=[0, 0, 1, 2],
        ppo_epochs=2,
        ppo_mini_batch_size=2,
        micro_batch_size=8,
    )
    assert _step_order(result) == [0, 1, 2, 0, 1, 2]
    assert result["zero_grad_count"] == 7
    assert result["optimizer_step_count"] == 6
    assert _rows_per_step(result) == [(0, 1), (2,), (3,), (0, 1), (2,), (3,)]
    assert result["events"][-1] == "zero_grad"
    assert result["events"][-2] == "optimizer_step"


@pytest.mark.parametrize("ppo_mini_batch_size", [1, 2, 8])
def test_ppo_mini_batch_size_never_defines_compression_boundary(monkeypatch, ppo_mini_batch_size):
    result = _run(
        monkeypatch,
        step_ids=[0, 0, 0, 1, 1],
        ppo_epochs=2,
        ppo_mini_batch_size=ppo_mini_batch_size,
        micro_batch_size=8,
    )
    assert _step_order(result) == [0, 1, 0, 1]
    assert _rows_per_step(result) == [(0, 1, 2), (3, 4), (0, 1, 2), (3, 4)]
    assert result["zero_grad_count"] == 5
    assert result["optimizer_step_count"] == 4
    assert result["events"][-1] == "zero_grad"


@pytest.mark.parametrize("micro_batch_size,expected_microbatches", [(1, 4), (2, 2), (4, 1)])
def test_microbatch_size_only_changes_memory_partition(monkeypatch, micro_batch_size, expected_microbatches):
    result = _run(
        monkeypatch,
        step_ids=[0, 0, 0, 0],
        ppo_epochs=1,
        ppo_mini_batch_size=1,
        micro_batch_size=micro_batch_size,
    )
    assert result["zero_grad_count"] == 2
    assert result["optimizer_step_count"] == 1
    assert len(result["optimizer_steps"][0]) == expected_microbatches
    assert _rows_per_step(result) == [(0, 1, 2, 3)]
    assert all(microbatch["step_ids"] == (0,) for microbatch in result["optimizer_steps"][0])
    assert result["events"][-1] == "zero_grad"
