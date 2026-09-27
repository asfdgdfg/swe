import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_weighted, kl_penalty


def test_trimmed_and_wide_weighted_objective_and_gradient_are_equal():
    old_wide = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0]])
    advantage_wide = torch.tensor([[1.0, -0.5, 0.25, 0.0, 0.0]])
    mask_wide = torch.tensor([[1, 0, 1, 0, 0]])
    weight_wide = torch.tensor([[0.5, 0.0, 0.5, 0.0, 0.0]])

    current_wide = torch.tensor(
        [[0.15, 0.25, 0.35, 99.0, -99.0]], requires_grad=True
    )
    wide = compute_policy_loss_weighted(
        old_log_prob=old_wide,
        log_prob=current_wide,
        advantages=advantage_wide,
        policy_mask=mask_wide,
        actor_token_weight=weight_wide,
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        clip_ratio_c=3.0,
    )[0]
    wide.backward()

    current_local = current_wide.detach()[:, :3].clone().requires_grad_(True)
    local = compute_policy_loss_weighted(
        old_log_prob=old_wide[:, :3],
        log_prob=current_local,
        advantages=advantage_wide[:, :3],
        policy_mask=mask_wide[:, :3],
        actor_token_weight=weight_wide[:, :3],
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        clip_ratio_c=3.0,
    )[0]
    local.backward()

    assert torch.allclose(wide, local)
    assert torch.allclose(current_wide.grad[:, :3], current_local.grad)
    assert torch.count_nonzero(current_wide.grad[:, 3:]) == 0


def test_trimmed_and_wide_entropy_and_kl_weighted_sums_are_equal():
    weights = torch.tensor([[0.4, 0.0, 0.6, 0.0, 0.0]])
    entropy = torch.tensor([[1.5, 2.0, 2.5, 100.0, 200.0]])
    current = torch.tensor([[0.2, 0.3, 0.4, 50.0, 60.0]])
    reference = torch.tensor([[0.1, 0.1, 0.2, -50.0, -60.0]])
    wide_entropy = torch.sum(entropy * weights)
    local_entropy = torch.sum(entropy[:, :3] * weights[:, :3])
    wide_kl = torch.sum(
        kl_penalty(current, reference, kl_penalty="k1") * weights
    )
    local_kl = torch.sum(
        kl_penalty(current[:, :3], reference[:, :3], kl_penalty="k1")
        * weights[:, :3]
    )
    assert torch.allclose(wide_entropy, local_entropy)
    assert torch.allclose(wide_kl, local_kl)


def test_realistic_zero_weight_logprob_overflow_is_numerically_neutral():
    result = compute_policy_loss_weighted(
        old_log_prob=torch.tensor([[-0.2, -100.0]]),
        log_prob=torch.tensor([[-0.1, 0.0]]),
        advantages=torch.tensor([[1.0, 1.0]]),
        policy_mask=torch.tensor([[1, 1]]),
        actor_token_weight=torch.tensor([[1.0, 0.0]]),
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        clip_ratio_c=3.0,
    )
    assert all(torch.isfinite(value) for value in result)
    single_active = compute_policy_loss_weighted(
        old_log_prob=torch.tensor([[-0.2]]),
        log_prob=torch.tensor([[-0.1]]),
        advantages=torch.tensor([[1.0]]),
        policy_mask=torch.tensor([[1]]),
        actor_token_weight=torch.tensor([[1.0]]),
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        clip_ratio_c=3.0,
    )
    for wide_value, active_value in zip(result, single_active, strict=True):
        assert torch.allclose(wide_value, active_value)


def test_extreme_active_policy_token_keeps_original_ppo_clipping():
    result = compute_policy_loss_weighted(
        old_log_prob=torch.tensor([[-100.0]]),
        log_prob=torch.tensor([[0.0]]),
        advantages=torch.tensor([[1.0]]),
        policy_mask=torch.tensor([[1]]),
        actor_token_weight=torch.tensor([[1.0]]),
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        clip_ratio_c=3.0,
    )
    pg_loss, clipfrac, ppo_kl, lower_clipfrac = result
    assert pg_loss.item() == pytest.approx(-1.2)
    assert clipfrac.item() == pytest.approx(1.0)
    assert ppo_kl.item() == pytest.approx(-100.0)
    assert lower_clipfrac.item() == pytest.approx(0.0)
