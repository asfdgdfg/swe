import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.workers.actor import dp_actor as dp_actor_module
from verl.workers.actor.dp_actor import DataParallelPPOActor, _trim_compression_microbatch


def _batch(compression=True):
    return DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[3, 4]]),
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "position_ids": torch.arange(4).view(1, -1),
            "old_log_probs": torch.zeros(1, 2),
            "advantages": torch.ones(1, 2),
            "policy_mask": torch.ones(1, 2, dtype=torch.long),
            "actor_token_weight": torch.full((1, 2), 0.5),
        },
        non_tensors={"optimizer_step_id": np.array([0])},
        meta_info={
            "temperature": 1.0,
            "compression_segment_training": compression,
            "compression_prompt_carrier_width": 2,
            "compression_response_carrier_width": 2,
        },
    )


def _actor():
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = torch.nn.Linear(1, 1)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.0)
    actor.use_ulysses_sp = True
    actor.use_remove_padding = False
    actor.ulysses_sequence_parallel_size = 2
    actor.config = OmegaConf.create(
        {
            "use_kl_loss": False,
            "use_dynamic_bsz": False,
            "ppo_epochs": 1,
            "ppo_mini_batch_size": 1,
            "ppo_micro_batch_size_per_gpu": 1,
            "ppo_max_token_len_per_gpu": 32,
            "clip_ratio": 0.2,
            "clip_ratio_low": None,
            "clip_ratio_high": None,
            "clip_ratio_c": 3.0,
            "entropy_coeff": 0.0,
            "loss_agg_mode": "token-mean",
            "kl_loss_coef": 0.0,
            "kl_loss_type": "k1",
            "grad_clip": 1.0,
        }
    )
    return actor


def test_compression_sp_guard_fails_before_forward_or_zero_grad(monkeypatch):
    actor = _actor()
    calls = {"forward": 0, "zero_grad": 0}
    actor._forward_micro_batch = lambda *a, **k: calls.__setitem__("forward", calls["forward"] + 1)
    original_zero = actor.actor_optimizer.zero_grad

    def zero_grad(*args, **kwargs):
        calls["zero_grad"] += 1
        return original_zero(*args, **kwargs)

    monkeypatch.setattr(actor.actor_optimizer, "zero_grad", zero_grad)
    update = inspect.unwrap(DataParallelPPOActor.update_policy)
    with pytest.raises(RuntimeError, match="requires use_remove_padding=True"):
        update(actor, _batch(True))
    assert calls == {"forward": 0, "zero_grad": 0}


def test_feature_false_does_not_trigger_compression_guard(monkeypatch):
    actor = _actor()
    forward_calls = 0
    monkeypatch.setattr(
        dp_actor_module,
        "get_torch_device",
        lambda: SimpleNamespace(current_device=lambda: "cpu"),
    )

    def forward(micro_batch, temperature, calculate_entropy):
        nonlocal forward_calls
        forward_calls += 1
        value = actor.actor_module.weight.sum().expand_as(micro_batch["old_log_probs"])
        return None, value

    actor._forward_micro_batch = forward
    monkeypatch.setattr(actor, "_optimizer_step", lambda: torch.tensor(0.0))
    update = inspect.unwrap(DataParallelPPOActor.update_policy)
    update(actor, _batch(False))
    assert forward_calls == 1


def test_same_sp_pair_rows_produce_identical_trimmed_values():
    batch = {
        "responses": torch.tensor([[30, 31, 0], [40, 41, 42]]),
        "input_ids": torch.tensor([[0, 10, 11, 30, 31, 0], [20, 21, 22, 40, 41, 42]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]),
        "position_ids": torch.tensor([[0, 0, 1, 2, 3, 0], [0, 1, 2, 3, 4, 5]]),
        "policy_mask": torch.tensor([[1, 0, 0], [1, 1, 1]]),
    }
    left, left_stats = _trim_compression_microbatch(batch, 3, 3)
    right, right_stats = _trim_compression_microbatch(
        {key: value.clone() for key, value in batch.items()}, 3, 3
    )
    assert left_stats == right_stats
    for key in ("input_ids", "attention_mask", "position_ids", "responses", "policy_mask"):
        assert torch.equal(left[key], right[key])
