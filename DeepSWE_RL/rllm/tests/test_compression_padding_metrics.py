import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.workers.actor import dp_actor as dp_actor_module
from verl.workers.actor.dp_actor import DataParallelPPOActor


def test_padding_metrics_match_hand_calculation(monkeypatch):
    monkeypatch.setattr(
        dp_actor_module,
        "get_torch_device",
        lambda: SimpleNamespace(current_device=lambda: "cpu"),
    )
    actor = object.__new__(DataParallelPPOActor)
    module = torch.nn.Linear(1, 1, bias=False)
    actor.actor_module = module
    actor.actor_optimizer = torch.optim.SGD(module.parameters(), lr=0.0)
    actor.use_ulysses_sp = False
    actor.use_remove_padding = True
    actor.ulysses_sequence_parallel_size = 1
    actor.config = OmegaConf.create(
        {
            "use_kl_loss": False, "use_dynamic_bsz": False, "ppo_epochs": 1,
            "ppo_mini_batch_size": 2, "ppo_micro_batch_size_per_gpu": 1,
            "ppo_max_token_len_per_gpu": 64, "clip_ratio": 0.2,
            "clip_ratio_low": None, "clip_ratio_high": None, "clip_ratio_c": 3.0,
            "entropy_coeff": 0.0, "loss_agg_mode": "token-mean",
            "kl_loss_coef": 0.0, "kl_loss_type": "k1", "grad_clip": 1.0,
        }
    )
    attention = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]]
    )
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[5, 6, 0, 0, 0], [7, 8, 9, 0, 0]]),
            "input_ids": torch.tensor([[0, 0, 1, 2, 5, 6, 0, 0, 0], [1, 2, 3, 4, 7, 8, 9, 0, 0]]),
            "attention_mask": attention,
            "position_ids": (attention.cumsum(-1) - 1) * attention,
            "old_log_probs": torch.zeros(2, 5), "advantages": torch.ones(2, 5),
            "policy_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]]),
            "actor_token_weight": torch.tensor([[0.2, 0.2, 0, 0, 0], [0.2, 0.2, 0.2, 0, 0]]),
        },
        non_tensors={"optimizer_step_id": np.array([0, 0])},
        meta_info={
            "temperature": 1.0, "compression_segment_training": True,
            "compression_prompt_carrier_width": 4,
            "compression_response_carrier_width": 5,
            "compression_max_prompt_length": 8,
            "compression_max_response_length": 10,
        },
    )

    def forward(micro_batch, temperature, calculate_entropy):
        value = module.weight.sum().expand_as(micro_batch["old_log_probs"])
        return None, value

    actor._forward_micro_batch = forward
    monkeypatch.setattr(actor, "_optimizer_step", lambda: torch.tensor(0.0))
    metrics = inspect.unwrap(DataParallelPPOActor.update_policy)(actor, batch)
    expected = {
        "compression/padding_tokens_before": 36,
        "compression/padding_tokens_after": 11,
        "compression/padding_tokens_saved": 25,
        "compression/padding_ratio_before": 25 / 36,
        "compression/padding_ratio_after": 0.0,
        "compression/batch_padding_tokens": 7,
        "compression/sp_divisibility_padding_tokens": 0,
        "compression/max_microbatch_prompt_len": 4,
        "compression/max_microbatch_response_len": 3,
        "compression/max_microbatch_total_len": 7,
    }
    for key, value in expected.items():
        assert metrics[key] == pytest.approx([value])
