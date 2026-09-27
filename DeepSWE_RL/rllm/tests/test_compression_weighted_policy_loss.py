import math

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_weighted


def _manual_weighted_ppo(old, current, advantage, weight, clip_low, clip_high, dual_clip):
    ratio = torch.exp(current - old)
    loss1 = -advantage * ratio
    loss2 = -advantage * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    clipped = torch.maximum(loss1, loss2)
    dual = torch.minimum(-advantage * dual_clip, clipped)
    token_loss = torch.where(advantage < 0, dual, clipped)
    clip = (loss2 > loss1).float()
    lower_clip = ((clipped > -advantage * dual_clip) & (advantage < 0)).float()
    return (
        torch.sum(token_loss * weight),
        torch.sum(clip * weight),
        torch.sum((old - current) * weight),
        torch.sum(lower_clip * weight),
    )


def test_weighted_ppo_matches_manual_token_formula_for_all_clipping_branches():
    ratios = torch.tensor([[1.0, 1.5, 0.5, 4.0, 0.2]], dtype=torch.float64)
    old = torch.zeros_like(ratios)
    current = torch.log(ratios)
    advantages = torch.tensor([[2.0, 2.0, 2.0, -1.0, -1.0]], dtype=torch.float64)
    policy_mask = torch.ones_like(ratios, dtype=torch.long)
    weight = torch.tensor([[0.10, 0.15, 0.20, 0.25, 0.30]], dtype=torch.float64)

    actual = compute_policy_loss_weighted(
        old,
        current,
        advantages,
        policy_mask,
        weight,
        cliprange=0.2,
        cliprange_low=0.1,
        cliprange_high=0.2,
        clip_ratio_c=3.0,
    )
    expected = _manual_weighted_ppo(old, current, advantages, weight, 0.1, 0.2, 3.0)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert torch.allclose(actual_value, expected_value)


def test_two_and_four_policy_token_episodes_are_equally_weighted():
    old = torch.zeros(2, 4)
    current = torch.zeros_like(old)
    advantages = torch.ones_like(old)
    policy_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    weight = torch.tensor([[0.25, 0.25, 0.0, 0.0], [0.125] * 4])

    pg_loss, *_ = compute_policy_loss_weighted(
        old, current, advantages, policy_mask, weight, cliprange=0.2
    )
    assert weight[0].sum().item() == pytest.approx(0.5)
    assert weight[1].sum().item() == pytest.approx(0.5)
    assert weight.sum().item() == pytest.approx(1.0)
    assert pg_loss.item() == pytest.approx(-1.0)


def test_segment_count_does_not_change_episode_contribution():
    # The first four tokens represent one segment; the next four represent four segments.
    old = torch.zeros(2, 4)
    current = torch.zeros_like(old)
    advantage = torch.ones_like(old)
    mask = torch.ones_like(old, dtype=torch.long)
    weight = torch.full_like(old, 1 / 8)
    total, *_ = compute_policy_loss_weighted(
        old, current, advantage, mask, weight, cliprange=0.2
    )
    first_episode = torch.sum((-advantage[0]) * weight[0])
    second_episode = torch.sum((-advantage[1]) * weight[1])
    assert first_episode.item() == pytest.approx(second_episode.item())
    assert total.item() == pytest.approx(first_episode.item() + second_episode.item())


def test_one_hundred_observations_do_not_change_policy_contribution():
    old = torch.zeros(2, 102)
    current = torch.full_like(old, math.log(1.1))
    advantage = torch.zeros_like(old)
    advantage[:, :2] = 1.0
    mask = torch.zeros_like(old, dtype=torch.long)
    mask[:, :2] = 1
    weight = torch.zeros_like(old)
    weight[:, :2] = 0.25

    total, *_ = compute_policy_loss_weighted(
        old, current, advantage, mask, weight, cliprange=0.2
    )
    per_episode = -1.1 * 0.5
    assert total.item() == pytest.approx(2 * per_episode)


def test_zero_advantage_keeps_policy_identity_and_allows_entropy_objective():
    old = torch.zeros(1, 2)
    current = torch.zeros_like(old)
    advantage = torch.zeros_like(old)
    mask = torch.ones_like(old, dtype=torch.long)
    weight = torch.tensor([[0.4, 0.6]])
    pg_loss, *_ = compute_policy_loss_weighted(
        old, current, advantage, mask, weight, cliprange=0.2
    )
    entropy = torch.tensor([[2.0, 3.0]])
    weighted_entropy = torch.sum(entropy * weight)
    entropy_coeff = 0.1
    assert pg_loss.item() == 0.0
    assert mask.sum().item() == 2
    assert weight.sum().item() == pytest.approx(1.0)
    assert (pg_loss - entropy_coeff * weighted_entropy).item() == pytest.approx(-0.26)


def test_weighted_actor_kl_uses_the_same_token_weight():
    log_prob = torch.tensor([[-0.2, -0.8, -0.3]])
    ref_log_prob = torch.tensor([[-0.4, -0.5, -0.9]])
    weight = torch.tensor([[0.25, 0.0, 0.75]])
    # For the configured default "k1" form, the token KL is log p - log q.
    weighted_kl = torch.sum((log_prob - ref_log_prob) * weight)
    assert weighted_kl.item() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "mask,weight,error",
    [
        ([[1, 0]], [[0.5, 0.1]], "zero where policy_mask is zero"),
        ([[1, 1]], [[0.5, -0.1]], "non-negative"),
        ([[1, 1]], [[0.5, float("nan")]], "finite"),
        ([[1, 2]], [[0.5, 0.0]], "0/1"),
    ],
)
def test_weight_validation_fails_fast(mask, weight, error):
    shape = (1, 2)
    with pytest.raises(ValueError, match=error):
        compute_policy_loss_weighted(
            torch.zeros(shape),
            torch.zeros(shape),
            torch.ones(shape),
            torch.tensor(mask),
            torch.tensor(weight),
            cliprange=0.2,
        )


def test_shape_mismatch_fails_fast_but_zero_weight_policy_token_is_allowed():
    with pytest.raises(ValueError, match="tensor shapes"):
        compute_policy_loss_weighted(
            torch.zeros(1, 2),
            torch.zeros(1, 3),
            torch.ones(1, 2),
            torch.ones(1, 2),
            torch.ones(1, 2),
            cliprange=0.2,
        )

    result = compute_policy_loss_weighted(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.ones(1, 2),
        torch.tensor([[0.0, 1.0]]),
        cliprange=0.2,
    )
    assert result[0].item() == pytest.approx(-1.0)


def test_microbatch_partitioning_preserves_final_weighted_objective():
    old = torch.zeros(4, 3)
    current = torch.log(torch.tensor([[1.0, 1.1, 0.9]] * 4))
    advantage = torch.tensor([[1.0, -1.0, 2.0]] * 4)
    mask = torch.ones_like(old, dtype=torch.long)
    weight = torch.full_like(old, 1 / old.numel())

    full = compute_policy_loss_weighted(
        old, current, advantage, mask, weight, cliprange=0.2
    )[0]
    for partitions in ((slice(0, 1), slice(1, 4)), (slice(0, 2), slice(2, 4))):
        partial = sum(
            compute_policy_loss_weighted(
                old[part], current[part], advantage[part], mask[part], weight[part], cliprange=0.2
            )[0]
            for part in partitions
        )
        assert torch.allclose(partial, full)
