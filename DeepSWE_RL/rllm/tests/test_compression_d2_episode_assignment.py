from rllm.trainer.verl.agent_ppo_trainer import (
    EpisodeRecord,
    _assign_episode_buckets_to_dp,
    _episode_compute_cost,
)


def _episode(source, lengths, group="g"):
    return EpisodeRecord(
        source, f"e{source}", group, 0.0, "done",
        [
            {"segment_id": i, "prompt_tokens": [1] * p, "response_tokens": [2] * r, "policy_mask": [1] * r}
            for i, (p, r) in enumerate(lengths)
        ],
        scalar_advantage=1.0,
    )


def test_cost_uses_unpadded_prompt_and_response_tokens():
    assert _episode_compute_cost(_episode(0, [(3, 4), (5, 6)])) == 18


def test_greedy_assignment_is_cost_descending_stable_and_deterministic():
    episodes = [_episode(0, [(1, 9)]), _episode(1, [(5, 3)]), _episode(2, [(4, 4)]), _episode(3, [(2, 4)])]
    result = _assign_episode_buckets_to_dp([episodes], 2)
    assert result.target_dp_by_source == {0: 0, 1: 1, 2: 1, 3: 0}
    assert result.dp_costs_by_step[0] == [16, 16]


def test_same_grpo_group_may_cross_dp_but_each_episode_is_atomic():
    episodes = [_episode(i, [(i + 1, 1), (1, 1)], group="same") for i in range(4)]
    result = _assign_episode_buckets_to_dp([episodes], 4)
    assert set(result.target_dp_by_source) == {0, 1, 2, 3}
    assert len(set(result.target_dp_by_source.values())) == 4
