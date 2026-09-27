import numpy as np
import torch

from rllm.trainer.verl.agent_ppo_trainer import (
    EpisodeRecord,
    _assign_episode_buckets_to_dp,
    _transform_policy_segments_d2,
)


def _episode(source, prompt_lengths):
    segments = []
    for segment_id, prompt_length in enumerate(prompt_lengths):
        segments.append(
            {
                "segment_id": segment_id,
                "prompt_tokens": list(range(10, 10 + prompt_length)),
                "response_tokens": [20 + segment_id, 30 + segment_id],
                "policy_mask": [1, 0],
            }
        )
    return EpisodeRecord(
        source_batch_idx=source,
        episode_id=f"episode-{source}",
        group_uid=f"group-{source // 2}",
        reward=float(source % 2),
        termination_reason="done",
        policy_segments=segments,
        scalar_advantage=0.25 * (source + 1),
    )


def test_production_d2_metadata_and_effective_weights_survive_local_carrier():
    episodes = [
        _episode(0, [2]),
        _episode(1, [5, 3]),
        _episode(2, [4]),
        _episode(3, [7, 2, 3]),
    ]
    buckets = [episodes]
    assignment = _assign_episode_buckets_to_dp(buckets, 2)
    step_by_source = {episode.source_batch_idx: 0 for episode in episodes}
    batch = _transform_policy_segments_d2(
        episodes,
        optimizer_step_by_source=step_by_source,
        assignment=assignment,
        effective_dp_degree=2,
        sp_size=2,
        eos_token_id=99,
        max_prompt_length=16,
        max_response_length=8,
        pad_token_id=0,
    )

    keys = (
        "episode_id", "group_uid", "segment_id", "source_batch_idx",
        "optimizer_step_id", "target_dp_replica", "is_dummy",
    )
    rows = [
        tuple(batch.non_tensor_batch[key][index] for key in keys)
        for index in range(len(batch))
    ]
    real_rows = [row for row in rows if not bool(row[-1])]
    expected_real = []
    for episode in episodes:
        for segment in episode.policy_segments:
            expected_real.append(
                (
                    episode.episode_id, episode.group_uid, segment["segment_id"],
                    episode.source_batch_idx, 0,
                    assignment.target_dp_by_source[episode.source_batch_idx], False,
                )
            )
    assert sorted(real_rows, key=str) == sorted(expected_real, key=str)
    assert all(int(row[4]) == 0 for row in rows)

    for index in np.where(~batch.non_tensor_batch["is_dummy"])[0]:
        policy = batch.batch["policy_mask"][index].bool()
        source = int(batch.non_tensor_batch["source_batch_idx"][index])
        episode = episodes[source]
        assert torch.all(
            batch.batch["advantages"][index][policy]
            == episode.scalar_advantage
        )
        expected_episode_weight = 1.0 / episode.policy_token_count
        assert torch.allclose(
            batch.batch["episode_normalized_weight"][index][policy],
            torch.full_like(
                batch.batch["episode_normalized_weight"][index][policy],
                expected_episode_weight,
            ),
        )
        expected_actor_weight = 2.0 / (len(episodes) * episode.policy_token_count)
        assert torch.allclose(
            batch.batch["actor_token_weight"][index][policy],
            torch.full_like(
                batch.batch["actor_token_weight"][index][policy],
                expected_actor_weight,
            ),
        )
    assert batch.meta_info["compression_prompt_carrier_width"] == 7
    assert batch.meta_info["compression_response_carrier_width"] == 2
