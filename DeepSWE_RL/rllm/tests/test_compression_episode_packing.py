import torch

from rllm.trainer.verl.agent_ppo_trainer import (
    EpisodeRecord,
    _pack_episode_optimizer_steps,
    _transform_policy_segments,
)


def _episode(index, segment_count=1):
    segments = [
        {
            "segment_id": segment_id,
            "prompt_tokens": [10 + index],
            "response_tokens": [20 + segment_id, 30 + segment_id],
            "policy_mask": [1, 1],
        }
        for segment_id in range(segment_count)
    ]
    return EpisodeRecord(index, f"e{index}", f"g{index // 2}", float(index % 2), "done", segments, 1.0)


def test_stable_episode_buckets_allow_smaller_tail():
    episodes = [_episode(index) for index in range(5)]
    assert [[e.source_batch_idx for e in b] for b in _pack_episode_optimizer_steps(episodes, 2)] == [
        [0, 1], [2, 3], [4]
    ]


def test_episode_segments_share_step_and_each_step_has_unit_logical_weight():
    episodes = [_episode(0, 1), _episode(1, 3), _episode(2, 2)]
    buckets = _pack_episode_optimizer_steps(episodes, 2)
    step_by_source = {
        episode.source_batch_idx: step_id
        for step_id, bucket in enumerate(buckets)
        for episode in bucket
    }
    data = _transform_policy_segments(
        episodes,
        max_prompt_length=4,
        max_response_length=4,
        pad_token_id=0,
        optimizer_step_by_source=step_by_source,
        effective_dp_degree=1,
    )
    for step_id in (0, 1):
        rows = data.non_tensor_batch["optimizer_step_id"] == step_id
        assert torch.isclose(data.batch["actor_token_weight"][rows].sum(), torch.tensor(1.0))
    assert list(data.non_tensor_batch["optimizer_step_id"]) == [0, 0, 0, 0, 1, 1]
