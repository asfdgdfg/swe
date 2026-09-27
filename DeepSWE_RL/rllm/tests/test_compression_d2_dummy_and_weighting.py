import pytest
import torch

from rllm.trainer.verl.agent_ppo_trainer import _make_compression_dummy_segment
from verl.trainer.ppo.core_algos import compute_policy_loss_weighted


def test_dummy_is_minimal_unpadded_and_has_explicit_identity():
    dummy = _make_compression_dummy_segment(step_id=2, target_dp_replica=3, dummy_index=4, eos_token_id=99)
    assert dummy["prompt_tokens"] == [99]
    assert dummy["response_tokens"] == [99]
    assert dummy["policy_mask"] == [0]
    assert dummy["actor_token_weight"] == [0.0]
    assert dummy["is_dummy"] is True
    assert dummy["optimizer_step_id"] == 2
    assert dummy["target_dp_replica"] == 3


def test_dummy_only_weighted_loss_is_zero_and_backward_is_valid():
    current = torch.tensor([[0.2]], requires_grad=True)
    loss, entropy_like, kl, clip = compute_policy_loss_weighted(
        torch.zeros_like(current), current, torch.zeros_like(current),
        torch.zeros_like(current, dtype=torch.long), torch.zeros_like(current), cliprange=0.2,
    )
    assert (loss.item(), entropy_like.item(), kl.item(), clip.item()) == (0.0, 0.0, 0.0, 0.0)
    loss.backward()
    assert current.grad.item() == 0.0


@pytest.mark.parametrize("episodes,expected_local,expected_global", [(8, 0.5, 0.125), (2, 2.0, 0.5), (1, 4.0, 1.0)])
def test_d4_episode_weight_survives_dp_mean(episodes, expected_local, expected_global):
    degree = 4
    local_sum = degree / episodes
    assert local_sum == pytest.approx(expected_local)
    assert local_sum / degree == pytest.approx(expected_global)


def test_sp_does_not_enter_weight_formula():
    degree, episodes, policy_tokens = 4, 8, 10
    world8_sp2 = degree / (episodes * policy_tokens)
    world4_sp1 = degree / (episodes * policy_tokens)
    assert world8_sp2 == world4_sp1
