import inspect

import numpy as np
import pytest
import torch

from rllm.trainer.verl.agent_ppo_trainer import (
    AgentPPOTrainer,
    EpisodeRecord,
    _build_episode_records,
    _compute_episode_policy_token_count,
    _extract_episode_scalar_advantages,
    _transform_policy_segments,
)


def _segment(segment_id, prompt, response, mask):
    return {
        "segment_id": segment_id,
        "prompt_tokens": prompt,
        "response_tokens": response,
        "policy_mask": mask,
    }


def _trajectory(idx, *, episode_id=None, reward=0.0, segments=None):
    return {
        "idx": idx,
        "episode_id": episode_id or f"episode-{idx}",
        "trajectory_reward": reward,
        "termination_reason": "done",
        "prompt_tokens": torch.tensor([10 + idx]),
        "response_tokens": torch.tensor([20 + idx]),
        "policy_segments": segments
        if segments is not None
        else [_segment(0, [10 + idx], [20 + idx], [1])],
    }


def _record(episode_id="episode-0", source_idx=0, segments=None, advantage=1.0):
    return EpisodeRecord(
        source_batch_idx=source_idx,
        episode_id=episode_id,
        group_uid="problem-0",
        reward=1.0,
        termination_reason="done",
        policy_segments=segments or [_segment(0, [11, 12], [21, 22, 23], [1, 0, 1])],
        scalar_advantage=advantage,
    )


def test_build_episode_records_preserves_explicit_identity_after_out_of_order_completion():
    uids = np.array(["p0"] * 4 + ["p1"] * 4, dtype=object)
    trajectories = [_trajectory(idx, reward=float(idx % 2)) for idx in [6, 0, 5, 3, 1, 7, 2, 4]]

    result = _build_episode_records(trajectories, uids)

    assert not result.skip_batch
    assert result.retained_source_batch_indices == list(range(8))
    assert [episode.source_batch_idx for episode in result.episodes] == list(range(8))
    assert [episode.episode_id for episode in result.episodes] == [f"episode-{i}" for i in range(8)]
    assert [episode.group_uid for episode in result.episodes] == ["p0"] * 4 + ["p1"] * 4


def test_missing_or_invalid_row_drops_the_complete_group():
    uids = np.array(["p0"] * 4 + ["p1"] * 4, dtype=object)
    missing = [_trajectory(idx) for idx in range(8) if idx != 2]
    result = _build_episode_records(missing, uids)
    assert result.invalid_source_batch_indices == [2]
    assert result.dropped_group_uids == ["p0"]
    assert result.retained_source_batch_indices == [4, 5, 6, 7]

    invalid = [_trajectory(idx) for idx in range(8)]
    invalid[5]["policy_segments"] = []
    result = _build_episode_records(invalid, uids)
    assert result.dropped_group_uids == ["p1"]
    assert result.retained_source_batch_indices == [0, 1, 2, 3]


@pytest.mark.parametrize("field,value", [("prompt_tokens", []), ("response_tokens", []), ("policy_segments", [])])
def test_empty_required_trajectory_data_invalidates_group(field, value):
    trajectories = [_trajectory(0), _trajectory(1)]
    trajectories[0][field] = value
    result = _build_episode_records(trajectories, np.array(["same", "same"], dtype=object))
    assert result.skip_batch
    assert result.dropped_group_uids == ["same"]


def test_identity_errors_fail_fast():
    uids = np.array(["p0", "p0"], dtype=object)
    with pytest.raises(ValueError, match="Duplicate trajectory source_batch_idx"):
        _build_episode_records([_trajectory(0), _trajectory(0, episode_id="other")], uids)
    with pytest.raises(ValueError, match="Duplicate trajectory episode_id"):
        _build_episode_records([_trajectory(0, episode_id="same"), _trajectory(1, episode_id="same")], uids)
    with pytest.raises(ValueError, match="outside repeated batch range"):
        _build_episode_records([_trajectory(2)], uids)


def test_extract_scalar_advantage_uses_only_valid_response_positions():
    episodes = [_record("e0", 3), _record("e1", 7)]
    advantages = torch.tensor([[99.0, 0.5, 0.5], [88.0, -1.25, -1.25]])
    response_mask = torch.tensor([[0, 1, 1], [0, 1, 1]])
    _extract_episode_scalar_advantages(episodes, advantages, response_mask, [3, 7])
    assert episodes[0].scalar_advantage == pytest.approx(0.5)
    assert episodes[1].scalar_advantage == pytest.approx(-1.25)


def test_extract_scalar_advantage_rejects_inconsistent_or_nonfinite_values():
    episode = _record()
    with pytest.raises(ValueError, match="consistent scalar advantage"):
        _extract_episode_scalar_advantages(
            [episode], torch.tensor([[1.0, 2.0]]), torch.ones(1, 2), [0]
        )
    with pytest.raises(ValueError, match="non-finite"):
        _extract_episode_scalar_advantages(
            [episode], torch.tensor([[float("nan")]]), torch.ones(1, 1), [0]
        )


def test_segment_transform_padding_masks_positions_and_metadata_are_exact():
    episode = _record(
        segments=[
            _segment(0, [11, 12], [21, 22, 23], [1, 0, 1]),
            _segment(1, [31], [41, 42], [0, 1]),
        ],
        advantage=-0.75,
    )
    batch = _transform_policy_segments(
        [episode], max_prompt_length=4, max_response_length=5, pad_token_id=0
    )

    assert batch.batch["prompts"].tolist() == [[11, 12], [0, 31]]
    assert batch.batch["responses"].tolist() == [[21, 22, 23], [41, 42, 0]]
    assert batch.batch["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ]
    assert batch.batch["position_ids"].tolist() == [
        [0, 1, 2, 3, 4],
        [0, 0, 1, 2, 0],
    ]
    assert batch.batch["policy_mask"].tolist() == [[1, 0, 1], [0, 1, 0]]
    assert batch.batch["advantages"].tolist() == [
        [-0.75, 0.0, -0.75],
        [0.0, -0.75, 0.0],
    ]
    assert batch.meta_info["compression_prompt_carrier_width"] == 2
    assert batch.meta_info["compression_response_carrier_width"] == 3
    assert batch.non_tensor_batch["episode_id"].tolist() == ["episode-0", "episode-0"]
    assert batch.non_tensor_batch["group_uid"].tolist() == ["problem-0", "problem-0"]
    assert batch.non_tensor_batch["segment_id"].tolist() == [0, 1]
    assert batch.non_tensor_batch["source_batch_idx"].tolist() == [0, 0]


def test_segment_transform_fails_instead_of_truncating_overflow():
    with pytest.raises(ValueError, match="prompt exceeds"):
        _transform_policy_segments(
            [_record(segments=[_segment(0, [1, 2, 3], [4], [1])])],
            max_prompt_length=2,
            max_response_length=2,
            pad_token_id=0,
        )
    with pytest.raises(ValueError, match="response exceeds"):
        _transform_policy_segments(
            [_record(segments=[_segment(0, [1], [2, 3, 4], [1, 1, 1])])],
            max_prompt_length=2,
            max_response_length=2,
            pad_token_id=0,
        )


def test_segment_schema_validation_reports_episode_metadata():
    duplicate = _record(segments=[_segment(0, [1], [2], [1]), _segment(0, [3], [4], [1])])
    with pytest.raises(ValueError, match="duplicate segment_id=0"):
        _compute_episode_policy_token_count(duplicate)

    mismatch = _record(segments=[_segment(0, [1], [2, 3], [1])])
    with pytest.raises(ValueError, match="response_tokens=2 but policy_mask=1"):
        _compute_episode_policy_token_count(mismatch)

    nonbinary = _record(segments=[_segment(0, [1], [2], [2])])
    with pytest.raises(ValueError, match="non-binary"):
        _compute_episode_policy_token_count(nonbinary)

    zero = _record(segments=[_segment(0, [1], [2, 3], [0, 0])])
    with pytest.raises(ValueError, match="episode_id='episode-0'.*termination_reason='done'"):
        _compute_episode_policy_token_count(zero)


def test_stage_7ca_helpers_are_not_wired_into_live_fit_or_rollout_return_shape():
    fit_source = inspect.getsource(AgentPPOTrainer.fit_agent)
    rollout_source = inspect.getsource(AgentPPOTrainer.generate_agent_trajectory)
    for helper_name in (
        "_build_episode_records",
        "_extract_episode_scalar_advantages",
        "_transform_policy_segments",
    ):
        assert helper_name not in fit_source
    assert "episode_normalized_weight" not in fit_source
    assert "return_raw" not in inspect.signature(AgentPPOTrainer.generate_agent_trajectory).parameters
    assert "policy_segments" not in rollout_source
