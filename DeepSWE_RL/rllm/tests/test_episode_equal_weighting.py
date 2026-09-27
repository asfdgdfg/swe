import numpy as np
import pytest
import torch

from rllm.trainer.verl.agent_ppo_trainer import (
    EpisodeRecord,
    _compute_rejection_sampling_selection,
    _extract_episode_scalar_advantages,
    _transform_policy_segments,
    compute_advantage,
)
from verl import DataProto
from verl.trainer.ppo.ray_trainer import AdvantageEstimator


def _segment(segment_id, response_len, policy_positions):
    mask = [1 if index in policy_positions else 0 for index in range(response_len)]
    return {
        "segment_id": segment_id,
        "prompt_tokens": [100 + segment_id],
        "response_tokens": list(range(200, 200 + response_len)),
        "policy_mask": mask,
    }


def _episode(index, group_uid, segments, scalar_advantage):
    return EpisodeRecord(
        source_batch_idx=index,
        episode_id=f"episode-{index}",
        group_uid=group_uid,
        reward=0.0,
        termination_reason="done",
        policy_segments=segments,
        scalar_advantage=scalar_advantage,
    )


def _episode_weight_sums(batch):
    result = {}
    for episode_id in np.unique(batch.non_tensor_batch["episode_id"]):
        rows = np.where(batch.non_tensor_batch["episode_id"] == episode_id)[0]
        result[episode_id] = batch.batch["episode_normalized_weight"][rows].sum().item()
    return result


def test_one_segment_and_five_segments_have_equal_episode_weight():
    episodes = [
        _episode(0, "p0", [_segment(0, 4, {0, 2})], 1.0),
        _episode(1, "p0", [_segment(i, 3, {0, 2}) for i in range(5)], -1.0),
    ]
    batch = _transform_policy_segments(
        episodes, max_prompt_length=3, max_response_length=5, pad_token_id=0
    )
    assert _episode_weight_sums(batch) == pytest.approx(
        {"episode-0": 1.0, "episode-1": 1.0}
    )
    assert episodes[0].policy_token_count == 2
    assert episodes[1].policy_token_count == 10


def test_requested_segment_and_token_count_extremes_each_have_unit_episode_weight():
    hundred_policy_tokens = _episode(0, "p0", [_segment(0, 100, set(range(100)))], 0.5)
    thirty_thirty_forty = _episode(
        1,
        "p0",
        [
            _segment(0, 30, set(range(30))),
            _segment(1, 30, set(range(30))),
            _segment(2, 40, set(range(40))),
        ],
        -0.5,
    )
    ten_policy_tokens = _episode(2, "p1", [_segment(0, 15, set(range(10)))], 0.25)
    thousand_policy_tokens = _episode(
        3, "p1", [_segment(0, 1000, set(range(1000)))], -0.25
    )
    batch = _transform_policy_segments(
        [hundred_policy_tokens, thirty_thirty_forty, ten_policy_tokens, thousand_policy_tokens],
        max_prompt_length=2,
        max_response_length=1000,
        pad_token_id=0,
    )
    assert _episode_weight_sums(batch) == pytest.approx(
        {"episode-0": 1.0, "episode-1": 1.0, "episode-2": 1.0, "episode-3": 1.0}
    )
    assert [episode.policy_token_count for episode in (
        hundred_policy_tokens,
        thirty_thirty_forty,
        ten_policy_tokens,
        thousand_policy_tokens,
    )] == [100, 100, 10, 1000]


def test_unequal_policy_token_counts_still_sum_to_one_per_episode():
    episodes = [
        _episode(0, "p0", [_segment(0, 2, {0})], 0.4),
        _episode(1, "p0", [_segment(0, 5, {0, 1, 3, 4})], -0.4),
    ]
    batch = _transform_policy_segments(
        episodes, max_prompt_length=2, max_response_length=5, pad_token_id=0
    )
    assert _episode_weight_sums(batch) == pytest.approx(
        {"episode-0": 1.0, "episode-1": 1.0}
    )
    assert batch.batch["episode_normalized_weight"][0].tolist() == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert batch.batch["episode_normalized_weight"][1].tolist() == pytest.approx(
        [0.25, 0.25, 0.0, 0.25, 0.25]
    )


def test_zero_advantage_remains_zero_without_removing_policy_weight():
    episode = _episode(0, "p0", [_segment(0, 4, {0, 2})], 0.0)
    batch = _transform_policy_segments(
        [episode], max_prompt_length=2, max_response_length=4, pad_token_id=0
    )
    assert torch.count_nonzero(batch.batch["advantages"]) == 0
    assert batch.batch["policy_mask"].sum().item() == 2
    assert batch.batch["episode_normalized_weight"].sum().item() == pytest.approx(1.0)


def test_real_grpo_advantage_is_computed_at_episode_level_before_expansion():
    # One GRPO problem group with four episode rewards [1, 0, 0, 1].
    response_mask = torch.tensor([[1, 1, 0]] * 4, dtype=torch.long)
    token_level_rewards = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    episode_batch = DataProto.from_dict(
        tensors={
            "responses": torch.ones(4, 3, dtype=torch.long),
            "response_mask": response_mask,
            "token_level_rewards": token_level_rewards,
        },
        non_tensors={"uid": np.array(["problem"] * 4, dtype=object)},
    )
    episode_batch = compute_advantage(
        episode_batch,
        adv_estimator=AdvantageEstimator.GRPO,
        norm_adv_by_std_in_grpo=True,
    )

    episodes = [
        _episode(
            index,
            "problem",
            [_segment(segment_id, 2 + segment_id, {0, 1}) for segment_id in range(index + 1)],
            0.0,
        )
        for index in range(4)
    ]
    _extract_episode_scalar_advantages(
        episodes,
        episode_batch.batch["advantages"],
        response_mask,
        [0, 1, 2, 3],
    )
    assert episodes[0].scalar_advantage == pytest.approx(episodes[3].scalar_advantage)
    assert episodes[1].scalar_advantage == pytest.approx(episodes[2].scalar_advantage)
    assert episodes[0].scalar_advantage > 0
    assert episodes[1].scalar_advantage < 0

    segment_batch = _transform_policy_segments(
        episodes, max_prompt_length=2, max_response_length=6, pad_token_id=0
    )
    assert len(segment_batch) == 10  # 1 + 2 + 3 + 4 segments, not ten GRPO episodes.
    assert _episode_weight_sums(segment_batch) == pytest.approx(
        {f"episode-{index}": 1.0 for index in range(4)}
    )
    for episode in episodes:
        rows = np.where(segment_batch.non_tensor_batch["episode_id"] == episode.episode_id)[0]
        policy_values = segment_batch.batch["advantages"][rows][
            segment_batch.batch["policy_mask"][rows].bool()
        ]
        assert torch.allclose(
            policy_values,
            torch.full_like(policy_values, episode.scalar_advantage),
        )


def test_rejection_helper_matches_existing_inline_semantics():
    uids = np.array(["none"] * 4 + ["all"] * 4 + ["partial"] * 4, dtype=object)
    per_episode_rewards = torch.tensor(
        [[0.0], [0.0], [-1.0], [0.0], [1.0], [2.0], [1.0], [3.0], [1.0], [0.0], [1.0], [0.0]]
    )

    expected_mask = torch.ones(len(uids), dtype=torch.bool)
    expected_none = expected_all = 0
    for uid in np.unique(uids):
        uid_mask = uids == uid
        rewards = per_episode_rewards[uid_mask].sum(-1)
        if (rewards <= 0).all():
            expected_mask[uid_mask] = False
            expected_none += 1
        elif (rewards >= 1).all():
            expected_mask[uid_mask] = False
            expected_all += 1

    actual = _compute_rejection_sampling_selection(uids, per_episode_rewards)
    assert torch.equal(actual.valid_mask, expected_mask)
    assert actual.solve_none == expected_none == 1
    assert actual.solve_all == expected_all == 1
    assert actual.solve_partial == 1


def test_zero_policy_tokens_fail_fast_with_traceable_metadata():
    episode = _episode(9, "problem-9", [_segment(0, 3, set())], 0.2)
    episode.termination_reason = "max_response_length_exceeded"
    with pytest.raises(
        ValueError,
        match="episode_id='episode-9'.*group_uid='problem-9'.*source_batch_idx=9.*max_response_length_exceeded",
    ):
        _transform_policy_segments(
            [episode], max_prompt_length=2, max_response_length=4, pad_token_id=0
        )
