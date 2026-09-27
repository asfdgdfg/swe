import asyncio
import json
import math
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import reduce
from pprint import pprint
from queue import Queue
from threading import Thread
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    _timer,
    compute_advantage,
    compute_data_metrics,
    compute_response_mask,
    compute_timing_metrics,
    reduce_metrics,
)

try:
    import r2egym
    from r2egym.agenthub.action import Action
    from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
except ImportError:
    raise ValueError("part1. import r2egym ERROR！！！！")

from rllm.environments.base.base_env import BaseEnv

try:
    R2EGYM_PATH = os.path.dirname(r2egym.__file__)
    print(f"R2EGYM_PATH: {R2EGYM_PATH}")
except Exception:
    raise ValueError("part2.find R2EGYM_PATH ERROR！！！！")


@dataclass
class EpisodeRecord:
    source_batch_idx: int
    episode_id: str
    group_uid: str
    reward: float
    termination_reason: str
    policy_segments: list[dict[str, Any]]
    scalar_advantage: float | None = None
    policy_token_count: int | None = None


@dataclass
class EpisodeRecordBuildResult:
    episodes: list[EpisodeRecord]
    retained_source_batch_indices: list[int]
    dropped_group_uids: list[str]
    invalid_source_batch_indices: list[int]
    skip_batch: bool


@dataclass
class RejectionSamplingSelection:
    valid_mask: torch.Tensor
    solve_none: int
    solve_all: int
    solve_partial: int


@dataclass
class AgentTrajectoryGenerationResult:
    trajectories: list[dict[str, Any]]
    transformed: DataProto
    metrics: dict[str, Any]


@dataclass
class DistributedEpisodeAssignment:
    target_dp_by_source: dict[int, int]
    episode_cost_by_source: dict[int, int]
    dp_costs_by_step: dict[int, list[int]]


def _pack_episode_optimizer_steps(
    episodes: list[EpisodeRecord], episode_mini_batch_size: int
) -> list[list[EpisodeRecord]]:
    if not isinstance(episode_mini_batch_size, int) or isinstance(
        episode_mini_batch_size, bool
    ) or episode_mini_batch_size <= 0:
        raise ValueError(
            "agent.compression_episode_mini_batch_size must be an integer greater than 0"
        )
    return [
        episodes[start : start + episode_mini_batch_size]
        for start in range(0, len(episodes), episode_mini_batch_size)
    ]


def _episode_compute_cost(episode: EpisodeRecord) -> int:
    return sum(
        len(segment["prompt_tokens"]) + len(segment["response_tokens"])
        for segment in episode.policy_segments
    )


def _assign_episode_buckets_to_dp(
    buckets: list[list[EpisodeRecord]], effective_dp_degree: int
) -> DistributedEpisodeAssignment:
    if effective_dp_degree <= 0:
        raise ValueError("effective_dp_degree must be positive")
    target_dp_by_source = {}
    episode_cost_by_source = {}
    dp_costs_by_step = {}
    for step_id, bucket in enumerate(buckets):
        dp_costs = [0] * effective_dp_degree
        ordered = sorted(
            bucket,
            key=lambda episode: (-_episode_compute_cost(episode), episode.source_batch_idx),
        )
        for episode in ordered:
            cost = _episode_compute_cost(episode)
            target_dp = min(range(effective_dp_degree), key=lambda dp: (dp_costs[dp], dp))
            target_dp_by_source[episode.source_batch_idx] = target_dp
            episode_cost_by_source[episode.source_batch_idx] = cost
            dp_costs[target_dp] += cost
        dp_costs_by_step[step_id] = dp_costs
    return DistributedEpisodeAssignment(
        target_dp_by_source=target_dp_by_source,
        episode_cost_by_source=episode_cost_by_source,
        dp_costs_by_step=dp_costs_by_step,
    )


def _make_compression_dummy_segment(
    *, step_id: int, target_dp_replica: int, dummy_index: int, eos_token_id: int
) -> dict[str, Any]:
    """Return a minimal, unpadded dummy representation independent of padding policy."""
    return {
        "episode_id": f"__compression_dummy__step{step_id}_dp{target_dp_replica}_row{dummy_index}",
        "group_uid": "__dummy__",
        "segment_id": -1,
        "source_batch_idx": -1,
        "optimizer_step_id": step_id,
        "target_dp_replica": target_dp_replica,
        "is_dummy": True,
        "prompt_tokens": [int(eos_token_id)],
        "response_tokens": [int(eos_token_id)],
        "policy_mask": [0],
        "scalar_advantage": 0.0,
        "episode_normalized_weight": [0.0],
        "actor_token_weight": [0.0],
    }


def _physical_rank_major_row_indices(
    rows: list[dict[str, Any]], *, effective_dp_degree: int, sp_size: int
) -> list[int]:
    """Lay out equal logical DP rows as equal physical-rank chunks."""
    if sp_size <= 0:
        raise ValueError("sp_size must be positive")
    step_ids = sorted({int(row["optimizer_step_id"]) for row in rows})
    per_physical_rank = [[] for _ in range(effective_dp_degree * sp_size)]
    for step_id in step_ids:
        dp_rows = []
        for dp in range(effective_dp_degree):
            indices = [
                index
                for index, row in enumerate(rows)
                if int(row["optimizer_step_id"]) == step_id
                and int(row["target_dp_replica"]) == dp
            ]
            dp_rows.append(indices)
        target_rows = {len(indices) for indices in dp_rows}
        if len(target_rows) != 1:
            raise ValueError(
                f"optimizer_step_id={step_id} has unequal logical DP row counts: "
                f"{[len(indices) for indices in dp_rows]}"
            )
        row_count = target_rows.pop()
        if row_count % sp_size != 0:
            raise ValueError(
                f"optimizer_step_id={step_id} target rows {row_count} are not divisible by SP={sp_size}"
            )
        local_rows = row_count // sp_size
        for dp, indices in enumerate(dp_rows):
            for sp_rank in range(sp_size):
                physical_rank = dp * sp_size + sp_rank
                start = sp_rank * local_rows
                per_physical_rank[physical_rank].extend(indices[start : start + local_rows])
    physical_lengths = {len(indices) for indices in per_physical_rank}
    if len(physical_lengths) != 1:
        raise ValueError(
            f"Physical-rank chunks must have equal lengths: "
            f"{[len(indices) for indices in per_physical_rank]}"
        )
    return [index for rank_indices in per_physical_rank for index in rank_indices]


def _select_unique_sp_result(results: Any, *, operation: str) -> DataProto:
    """Collapse ONE_TO_ALL SP views without concatenating logical rows."""
    if isinstance(results, DataProto):
        return results
    if not isinstance(results, (list, tuple)) or not results:
        raise ValueError(f"{operation} returned no SP worker results")
    if not all(isinstance(result, DataProto) for result in results):
        raise TypeError(f"{operation} must return DataProto from every SP rank")
    canonical = results[0]
    canonical_keys = list(canonical.batch.keys())
    canonical_rows = (
        next(iter(canonical.batch.values())).shape[0] if canonical_keys else None
    )
    for rank, result in enumerate(results[1:], start=1):
        result_keys = list(result.batch.keys())
        result_rows = (
            next(iter(result.batch.values())).shape[0] if result_keys else None
        )
        if result_rows != canonical_rows:
            raise ValueError(
                f"{operation} changed logical row count across SP ranks: "
                f"rank0={canonical_rows}, rank{rank}={result_rows}"
            )
        if set(result.batch.keys()) != set(canonical.batch.keys()):
            raise ValueError(f"{operation} returned inconsistent tensor fields across SP ranks")
        for key in canonical.batch.keys():
            if canonical.batch[key].shape != result.batch[key].shape:
                raise ValueError(
                    f"{operation} returned inconsistent {key} shape across SP ranks"
                )
            if not torch.allclose(
                canonical.batch[key].to(torch.float32),
                result.batch[key].to(torch.float32),
                rtol=1e-5,
                atol=1e-7,
            ):
                raise ValueError(
                    f"{operation} returned inconsistent {key} values across SP ranks"
                )
    return canonical


def _build_episode_advantage_batch(episodes: list[EpisodeRecord], pad_token_id: int) -> DataProto:
    """Represent each real episode by one reward-bearing token for GRPO."""
    if not episodes:
        raise ValueError("Cannot compute episode-level advantages for an empty episode list")
    rewards = torch.tensor([episode.reward for episode in episodes], dtype=torch.float32).unsqueeze(-1)
    tokens = torch.full((len(episodes), 1), int(pad_token_id), dtype=torch.long)
    return DataProto.from_dict(
        tensors={
            "prompts": tokens.clone(),
            "responses": tokens.clone(),
            "input_ids": torch.cat([tokens, tokens], dim=-1),
            "attention_mask": torch.ones((len(episodes), 2), dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long).repeat(len(episodes), 1),
            "token_level_scores": rewards.clone(),
            "token_level_rewards": rewards,
        },
        non_tensors={
            "uid": np.array([episode.group_uid for episode in episodes], dtype=object),
            "source_batch_idx": np.array([episode.source_batch_idx for episode in episodes]),
        },
    )


def _tensor_numel(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel()
    try:
        return len(value)
    except TypeError:
        return 0


def _build_episode_records(
    trajectories: list[dict[str, Any]],
    repeated_group_uids: np.ndarray,
) -> EpisodeRecordBuildResult:
    """Bind rollout results to their explicit repeated-batch rows.

    Missing or invalid rollouts invalidate their complete GRPO group. This is a
    pure preparation helper and is deliberately not connected to the live actor
    path in stage 7C-A.
    """
    expected_num_rows = len(repeated_group_uids)
    expected_indices = set(range(expected_num_rows))
    trajectories_by_idx: dict[int, dict[str, Any]] = {}
    episode_ids: set[str] = set()
    invalid_source_indices: set[int] = set()

    for trajectory in trajectories:
        if "idx" not in trajectory:
            raise ValueError("Trajectory is missing required source batch field 'idx'")
        source_batch_idx = int(trajectory["idx"])
        if source_batch_idx < 0 or source_batch_idx >= expected_num_rows:
            raise ValueError(
                f"Trajectory source_batch_idx={source_batch_idx} is outside repeated batch range "
                f"[0, {expected_num_rows})"
            )
        if source_batch_idx in trajectories_by_idx:
            raise ValueError(f"Duplicate trajectory source_batch_idx={source_batch_idx}")
        trajectories_by_idx[source_batch_idx] = trajectory

        episode_id = trajectory.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            invalid_source_indices.add(source_batch_idx)
        elif episode_id in episode_ids:
            raise ValueError(f"Duplicate trajectory episode_id={episode_id!r}")
        else:
            episode_ids.add(episode_id)

        policy_segments = trajectory.get("policy_segments")
        if (
            _tensor_numel(trajectory.get("prompt_tokens")) == 0
            or _tensor_numel(trajectory.get("response_tokens")) == 0
            or "trajectory_reward" not in trajectory
            or not isinstance(policy_segments, list)
            or len(policy_segments) == 0
        ):
            invalid_source_indices.add(source_batch_idx)

    missing_indices = expected_indices - set(trajectories_by_idx)
    invalid_source_indices.update(missing_indices)
    invalid_group_uids = {
        str(repeated_group_uids[source_batch_idx])
        for source_batch_idx in invalid_source_indices
    }

    retained_source_indices = [
        source_batch_idx
        for source_batch_idx in range(expected_num_rows)
        if str(repeated_group_uids[source_batch_idx]) not in invalid_group_uids
    ]

    episodes = []
    for source_batch_idx in retained_source_indices:
        trajectory = trajectories_by_idx[source_batch_idx]
        episodes.append(
            EpisodeRecord(
                source_batch_idx=source_batch_idx,
                episode_id=trajectory["episode_id"],
                group_uid=str(repeated_group_uids[source_batch_idx]),
                reward=float(trajectory["trajectory_reward"]),
                termination_reason=str(trajectory.get("termination_reason", "")),
                policy_segments=trajectory["policy_segments"],
            )
        )

    return EpisodeRecordBuildResult(
        episodes=episodes,
        retained_source_batch_indices=retained_source_indices,
        dropped_group_uids=sorted(invalid_group_uids),
        invalid_source_batch_indices=sorted(invalid_source_indices),
        skip_batch=len(episodes) == 0,
    )


def _compute_rejection_sampling_selection(
    uids: np.ndarray,
    reward_tensor: torch.Tensor,
) -> RejectionSamplingSelection:
    """Mirror the current AgentPPOTrainer episode-level rejection semantics."""
    unique_uids = np.unique(uids)
    valid_mask = torch.ones(len(uids), dtype=torch.bool)
    solve_none = 0
    solve_all = 0
    for uid in unique_uids:
        uid_mask = uids == uid
        uid_rewards = reward_tensor[uid_mask].sum(-1)
        if (uid_rewards <= 0).all():
            valid_mask[uid_mask] = False
            solve_none += 1
        elif (uid_rewards >= 1).all():
            valid_mask[uid_mask] = False
            solve_all += 1
    return RejectionSamplingSelection(
        valid_mask=valid_mask,
        solve_none=solve_none,
        solve_all=solve_all,
        solve_partial=len(unique_uids) - solve_none - solve_all,
    )


def _extract_episode_scalar_advantages(
    episodes: list[EpisodeRecord],
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    row_source_batch_indices: list[int] | np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> None:
    """Attach one verified scalar GRPO advantage to each episode record."""
    if advantages.shape != response_mask.shape:
        raise ValueError(
            f"advantages shape {tuple(advantages.shape)} does not match response_mask "
            f"shape {tuple(response_mask.shape)}"
        )
    if advantages.shape[0] != len(row_source_batch_indices):
        raise ValueError("Advantage rows do not match source_batch_idx rows")

    episodes_by_source = {episode.source_batch_idx: episode for episode in episodes}
    if len(episodes_by_source) != len(episodes):
        raise ValueError("Episode records contain duplicate source_batch_idx values")

    attached_sources = set()
    for row_idx, source_batch_idx_value in enumerate(row_source_batch_indices):
        source_batch_idx = int(source_batch_idx_value)
        episode = episodes_by_source.get(source_batch_idx)
        if episode is None:
            raise ValueError(f"No EpisodeRecord for advantage source_batch_idx={source_batch_idx}")
        if source_batch_idx in attached_sources:
            raise ValueError(f"Duplicate advantage row for source_batch_idx={source_batch_idx}")

        valid_values = advantages[row_idx][response_mask[row_idx].bool()]
        if valid_values.numel() == 0:
            raise ValueError(f"Episode {episode.episode_id!r} has no valid response advantage positions")
        if not torch.isfinite(valid_values).all():
            raise ValueError(f"Episode {episode.episode_id!r} has non-finite advantages")
        reference_value = valid_values[0]
        if not torch.allclose(
            valid_values,
            torch.full_like(valid_values, reference_value),
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(
                f"Episode {episode.episode_id!r} does not have one consistent scalar advantage: "
                f"{valid_values.tolist()}"
            )
        episode.scalar_advantage = float(reference_value.item())
        attached_sources.add(source_batch_idx)

    missing_sources = set(episodes_by_source) - attached_sources
    if missing_sources:
        raise ValueError(f"Missing advantage rows for source_batch_idx={sorted(missing_sources)}")


def _compute_episode_policy_token_count(episode: EpisodeRecord) -> int:
    """Validate Stage 6 segments and attach the episode-wide policy-token count."""
    segment_ids = set()
    policy_token_count = 0
    for segment in episode.policy_segments:
        segment_id = int(segment["segment_id"])
        if segment_id in segment_ids:
            raise ValueError(
                f"Episode {episode.episode_id!r} has duplicate segment_id={segment_id}"
            )
        segment_ids.add(segment_id)

        response_tokens = segment["response_tokens"]
        policy_mask = segment["policy_mask"]
        if len(response_tokens) != len(policy_mask):
            raise ValueError(
                f"Episode {episode.episode_id!r} segment {segment_id} has "
                f"response_tokens={len(response_tokens)} but policy_mask={len(policy_mask)}"
            )
        invalid_mask_values = [value for value in policy_mask if value not in (0, 1)]
        if invalid_mask_values:
            raise ValueError(
                f"Episode {episode.episode_id!r} segment {segment_id} has non-binary "
                f"policy_mask values: {invalid_mask_values[:5]}"
            )
        policy_token_count += sum(int(value) for value in policy_mask)

    if policy_token_count == 0:
        raise ValueError(
            "Episode has zero policy tokens: "
            f"episode_id={episode.episode_id!r}, group_uid={episode.group_uid!r}, "
            f"source_batch_idx={episode.source_batch_idx}, "
            f"termination_reason={episode.termination_reason!r}, "
            f"segment_count={len(episode.policy_segments)}"
        )
    episode.policy_token_count = policy_token_count
    return policy_token_count


def _transform_policy_segments(
    episodes: list[EpisodeRecord],
    *,
    max_prompt_length: int,
    max_response_length: int,
    pad_token_id: int,
    optimizer_step_by_source: dict[int, int] | None = None,
    effective_dp_degree: int | None = None,
) -> DataProto:
    """Create the CPU segment DataProto used by stage 7C-A structural tests."""
    raw_segments = [
        segment
        for episode in episodes
        for segment in episode.policy_segments
    ]
    if not raw_segments:
        raise ValueError("Cannot transform an empty list of policy segments")

    # The configured lengths are hard limits, not tensor target widths.  The
    # rectangular DataProto carrier only needs to be as wide as this packed
    # segment batch; the actor trims it again for each forward microbatch.
    prompt_carrier_width = max(len(segment["prompt_tokens"]) for segment in raw_segments)
    response_carrier_width = max(len(segment["response_tokens"]) for segment in raw_segments)
    segment_rows = []
    for episode in episodes:
        if episode.scalar_advantage is None:
            raise ValueError(f"Episode {episode.episode_id!r} is missing scalar_advantage")
        policy_token_count = _compute_episode_policy_token_count(episode)
        optimizer_step_id = (
            optimizer_step_by_source.get(episode.source_batch_idx)
            if optimizer_step_by_source is not None
            else None
        )
        for segment in episode.policy_segments:
            prompt_tokens = list(segment["prompt_tokens"])
            response_tokens = list(segment["response_tokens"])
            policy_mask = [int(value) for value in segment["policy_mask"]]
            segment_id = int(segment["segment_id"])

            if len(prompt_tokens) > max_prompt_length:
                raise ValueError(
                    f"Segment prompt exceeds configured limit: episode_id={episode.episode_id!r}, "
                    f"segment_id={segment_id}, actual_prompt_length={len(prompt_tokens)}, "
                    f"max_prompt_length={max_prompt_length}"
                )
            if len(response_tokens) > max_response_length:
                raise ValueError(
                    f"Segment response exceeds configured limit: episode_id={episode.episode_id!r}, "
                    f"segment_id={segment_id}, actual_response_length={len(response_tokens)}, "
                    f"max_response_length={max_response_length}"
                )

            padded_prompt = [pad_token_id] * (prompt_carrier_width - len(prompt_tokens)) + prompt_tokens
            padded_response = response_tokens + [pad_token_id] * (response_carrier_width - len(response_tokens))
            prompt_attention = [0] * (prompt_carrier_width - len(prompt_tokens)) + [1] * len(prompt_tokens)
            response_attention = [1] * len(response_tokens) + [0] * (response_carrier_width - len(response_tokens))
            padded_policy_mask = policy_mask + [0] * (response_carrier_width - len(policy_mask))

            policy_mask_tensor = torch.tensor(padded_policy_mask, dtype=torch.long)
            segment_rows.append(
                {
                    "episode": episode,
                    "segment_id": segment_id,
                    "prompts": torch.tensor(padded_prompt, dtype=torch.long),
                    "responses": torch.tensor(padded_response, dtype=torch.long),
                    "attention_mask": torch.tensor(prompt_attention + response_attention, dtype=torch.long),
                    "policy_mask": policy_mask_tensor,
                    "advantages": policy_mask_tensor.to(torch.float32) * episode.scalar_advantage,
                    "episode_normalized_weight": policy_mask_tensor.to(torch.float32) / policy_token_count,
                    "actor_token_weight": (
                        policy_mask_tensor.to(torch.float32) / policy_token_count
                        if optimizer_step_by_source is not None
                        else None
                    ),
                    "optimizer_step_id": optimizer_step_id,
                }
            )

    if not segment_rows:
        raise ValueError("Cannot transform an empty list of policy segments")

    tensor_batch = {
        "prompts": torch.stack([row["prompts"] for row in segment_rows]),
        "responses": torch.stack([row["responses"] for row in segment_rows]),
        "attention_mask": torch.stack([row["attention_mask"] for row in segment_rows]),
        "policy_mask": torch.stack([row["policy_mask"] for row in segment_rows]),
        "advantages": torch.stack([row["advantages"] for row in segment_rows]),
        "episode_normalized_weight": torch.stack(
            [row["episode_normalized_weight"] for row in segment_rows]
        ),
    }
    if optimizer_step_by_source is not None:
        tensor_batch["actor_token_weight"] = torch.stack(
            [row["actor_token_weight"] for row in segment_rows]
        )
    tensor_batch["input_ids"] = torch.cat(
        [tensor_batch["prompts"], tensor_batch["responses"]], dim=-1
    )
    tensor_batch["position_ids"] = (
        torch.cumsum(tensor_batch["attention_mask"], dim=-1) - 1
    ) * tensor_batch["attention_mask"]

    non_tensor_batch = {
        "episode_id": np.array(
            [row["episode"].episode_id for row in segment_rows], dtype=object
        ),
        "group_uid": np.array(
            [row["episode"].group_uid for row in segment_rows], dtype=object
        ),
        "segment_id": np.array([row["segment_id"] for row in segment_rows]),
        "source_batch_idx": np.array(
            [row["episode"].source_batch_idx for row in segment_rows]
        ),
    }
    if optimizer_step_by_source is not None:
        non_tensor_batch["optimizer_step_id"] = np.array(
            [row["optimizer_step_id"] for row in segment_rows], dtype=np.int64
        )
    segment_batch = DataProto.from_dict(
        tensors=tensor_batch,
        non_tensors=non_tensor_batch,
        meta_info={
            "compression_segment_training_stage": "7C-D1" if optimizer_step_by_source is not None else "7C-A",
            "compression_prompt_carrier_width": prompt_carrier_width,
            "compression_response_carrier_width": response_carrier_width,
            "compression_max_prompt_length": max_prompt_length,
            "compression_max_response_length": max_response_length,
            "compression_padding_tokens_before": len(segment_rows)
            * (max_prompt_length + max_response_length),
            "compression_padding_tokens_carrier": len(segment_rows)
            * (prompt_carrier_width + response_carrier_width),
            **(
                {
                    "compression_segment_training": True,
                    "effective_dp_degree": effective_dp_degree,
                }
                if optimizer_step_by_source is not None
                else {}
            ),
        },
    )

    for episode in episodes:
        episode_rows = np.where(
            segment_batch.non_tensor_batch["episode_id"] == episode.episode_id
        )[0]
        total_weight = segment_batch.batch["episode_normalized_weight"][episode_rows].sum()
        if not torch.isclose(total_weight, torch.tensor(1.0), rtol=1e-5, atol=1e-7):
            raise ValueError(
                f"Episode {episode.episode_id!r} normalized weight sums to "
                f"{total_weight.item()} instead of 1"
            )
    if optimizer_step_by_source is not None:
        for step_id in sorted(set(optimizer_step_by_source.values())):
            source_indices = [
                episode.source_batch_idx
                for episode in episodes
                if optimizer_step_by_source[episode.source_batch_idx] == step_id
            ]
            episode_count = len(source_indices)
            step_rows = np.isin(
                segment_batch.non_tensor_batch["source_batch_idx"], source_indices
            )
            dp_scale = float(effective_dp_degree or 1)
            segment_batch.batch["actor_token_weight"][step_rows] *= dp_scale / episode_count
            logical_weight = segment_batch.batch["actor_token_weight"][step_rows].sum()
            if not torch.isclose(logical_weight, torch.tensor(dp_scale), rtol=1e-5, atol=1e-7):
                raise ValueError(
                    f"optimizer_step_id={step_id} logical actor weight sums to "
                    f"{logical_weight.item()} instead of effective DP degree {dp_scale}"
                )
    return segment_batch


def _transform_policy_segments_d2(
    episodes: list[EpisodeRecord],
    *,
    optimizer_step_by_source: dict[int, int],
    assignment: DistributedEpisodeAssignment,
    effective_dp_degree: int,
    sp_size: int,
    eos_token_id: int,
    max_prompt_length: int,
    max_response_length: int,
    pad_token_id: int,
) -> DataProto:
    """Add D2 assignment/dummies and create a physical-rank-major DataProto."""
    real_batch = _transform_policy_segments(
        episodes,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        pad_token_id=pad_token_id,
        optimizer_step_by_source=optimizer_step_by_source,
        effective_dp_degree=effective_dp_degree,
    )
    real_sources = real_batch.non_tensor_batch["source_batch_idx"]
    real_batch.non_tensor_batch["target_dp_replica"] = np.array(
        [assignment.target_dp_by_source[int(source)] for source in real_sources],
        dtype=np.int64,
    )
    real_batch.non_tensor_batch["is_dummy"] = np.zeros(len(real_batch), dtype=bool)
    prompt_carrier_width = int(
        real_batch.meta_info["compression_prompt_carrier_width"]
    )
    response_carrier_width = int(
        real_batch.meta_info["compression_response_carrier_width"]
    )

    layout_rows = [
        {
            "optimizer_step_id": int(real_batch.non_tensor_batch["optimizer_step_id"][index]),
            "target_dp_replica": int(real_batch.non_tensor_batch["target_dp_replica"][index]),
            "data_index": index,
            "is_dummy": False,
        }
        for index in range(len(real_batch))
    ]
    dummy_specs = []
    step_ids = sorted(set(optimizer_step_by_source.values()))
    for step_id in step_ids:
        real_rows = [
            sum(
                row["optimizer_step_id"] == step_id
                and row["target_dp_replica"] == dp
                for row in layout_rows
            )
            for dp in range(effective_dp_degree)
        ]
        max_rows = max(real_rows)
        target_rows = math.ceil(max_rows / sp_size) * sp_size
        if target_rows == 0:
            raise ValueError(f"optimizer_step_id={step_id} contains no real segment rows")
        for dp, row_count in enumerate(real_rows):
            for dummy_index in range(target_rows - row_count):
                dummy_specs.append(
                    _make_compression_dummy_segment(
                        step_id=step_id,
                        target_dp_replica=dp,
                        dummy_index=dummy_index,
                        eos_token_id=eos_token_id,
                    )
                )

    if dummy_specs:
        dummy_count = len(dummy_specs)
        prompts = torch.full(
            (dummy_count, prompt_carrier_width), pad_token_id, dtype=torch.long
        )
        responses = torch.full(
            (dummy_count, response_carrier_width), pad_token_id, dtype=torch.long
        )
        prompts[:, -1] = eos_token_id
        responses[:, 0] = eos_token_id
        attention_mask = torch.zeros(
            (dummy_count, prompt_carrier_width + response_carrier_width), dtype=torch.long
        )
        attention_mask[:, prompt_carrier_width - 1] = 1
        attention_mask[:, prompt_carrier_width] = 1
        zeros = torch.zeros((dummy_count, response_carrier_width), dtype=torch.float32)
        dummy_batch = DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
                "input_ids": torch.cat([prompts, responses], dim=-1),
                "attention_mask": attention_mask,
                "position_ids": (
                    torch.cumsum(attention_mask, dim=-1) - 1
                ) * attention_mask,
                "policy_mask": zeros.to(torch.long),
                "advantages": zeros.clone(),
                "episode_normalized_weight": zeros.clone(),
                "actor_token_weight": zeros.clone(),
            },
            non_tensors={
                "episode_id": np.array([spec["episode_id"] for spec in dummy_specs], dtype=object),
                "group_uid": np.array([spec["group_uid"] for spec in dummy_specs], dtype=object),
                "segment_id": np.full(dummy_count, -1, dtype=np.int64),
                "source_batch_idx": np.full(dummy_count, -1, dtype=np.int64),
                "optimizer_step_id": np.array([spec["optimizer_step_id"] for spec in dummy_specs], dtype=np.int64),
                "target_dp_replica": np.array([spec["target_dp_replica"] for spec in dummy_specs], dtype=np.int64),
                "is_dummy": np.ones(dummy_count, dtype=bool),
            },
            meta_info=real_batch.meta_info,
        )
        combined = DataProto.concat([real_batch, dummy_batch])
    else:
        combined = real_batch

    layout_rows = [
        {
            "optimizer_step_id": int(combined.non_tensor_batch["optimizer_step_id"][index]),
            "target_dp_replica": int(combined.non_tensor_batch["target_dp_replica"][index]),
        }
        for index in range(len(combined))
    ]
    physical_indices = _physical_rank_major_row_indices(
        layout_rows,
        effective_dp_degree=effective_dp_degree,
        sp_size=sp_size,
    )
    combined = combined.select_idxs(physical_indices)
    combined.meta_info.update(
        {
            "compression_segment_training_stage": "7C-D2",
            "effective_dp_degree": effective_dp_degree,
            "ulysses_sequence_parallel_size": sp_size,
            "real_segment_rows": len(real_batch),
            "dummy_rows": len(dummy_specs),
            "dummy_tokens": len(dummy_specs) * 2,
            "dp_costs_by_step": assignment.dp_costs_by_step,
            "compression_padding_tokens_before": len(combined)
            * (max_prompt_length + max_response_length),
            "compression_padding_tokens_carrier": len(combined)
            * (prompt_carrier_width + response_carrier_width),
        }
    )
    return combined


class AgentPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
    ):
        print("===="*100)
        import verl.trainer.ppo.ray_trainer as rt
        print(rt.__file__)
        print("===="*100)
        super().__init__(config=config, tokenizer=tokenizer, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls, reward_fn=reward_fn, val_reward_fn=val_reward_fn)
        self.env_class = env_class
        self.agent_class = agent_class
        self.env_args = env_args or {}
        self.agent_args = agent_args or {}

        if self.config.agent.use_stepwise_advantage:
            print("Using step-level advantage, max_prompt_length and max_response_length will be applied step-wise")
        else:
            print("Using trajectory-level advantage, max_prompt_length and max_response_length will be applied episode-wise")

    def _query_compression_effective_dp_degree(self) -> int:
        values = self.actor_rollout_wg.get_effective_data_parallel_size()
        if not isinstance(values, (list, tuple)):
            values = [values]
        normalized = [int(value) for value in values]
        if not normalized or len(set(normalized)) != 1:
            raise RuntimeError(
                f"FSDP workers disagree on effective data parallel degree: {normalized}"
            )
        return normalized[0]

    def _validate_compression_d1_config(self, effective_dp_degree: int) -> int:
        if effective_dp_degree != 1:
            raise RuntimeError(
                "Compression segment training stage 7C-D1 currently requires effective DP degree == 1"
            )
        if self.config.actor_rollout_ref.actor.use_dynamic_bsz:
            raise RuntimeError(
                "compression segment training stage 7C-D1 currently requires use_dynamic_bsz=False"
            )
        episode_mini_batch_size = self.config.agent.get(
            "compression_episode_mini_batch_size", None
        )
        if not isinstance(episode_mini_batch_size, int) or isinstance(
            episode_mini_batch_size, bool
        ) or episode_mini_batch_size <= 0:
            raise ValueError(
                "agent.compression_episode_mini_batch_size must be an integer greater than 0"
            )
        if self.config.agent.use_stepwise_advantage:
            raise ValueError("Context compression training requires trajectory-level advantage")
        return episode_mini_batch_size

    def _validate_compression_config(self, effective_dp_degree: int) -> int:
        if effective_dp_degree <= 0:
            raise RuntimeError("Compression training requires a positive effective DP degree")
        if effective_dp_degree == 1:
            return self._validate_compression_d1_config(effective_dp_degree)
        if self.config.actor_rollout_ref.actor.use_dynamic_bsz:
            raise RuntimeError(
                "compression segment training currently requires use_dynamic_bsz=False"
            )
        episode_mini_batch_size = self.config.agent.get(
            "compression_episode_mini_batch_size", None
        )
        if not isinstance(episode_mini_batch_size, int) or isinstance(
            episode_mini_batch_size, bool
        ) or episode_mini_batch_size <= 0:
            raise ValueError(
                "agent.compression_episode_mini_batch_size must be an integer greater than 0"
            )
        if self.config.agent.use_stepwise_advantage:
            raise ValueError("Context compression training requires trajectory-level advantage")
        return episode_mini_batch_size

    def _run_compression_training_step(
        self,
        *,
        batch: DataProto,
        timing_raw: dict[str, Any],
        effective_dp_degree: int,
        episode_mini_batch_size: int,
    ) -> tuple[DataProto | None, dict[str, Any]]:
        raw_result = self.generate_agent_trajectory(
            timing_raw=timing_raw,
            meta_info=batch.meta_info,
            return_raw_trajectories=True,
        )
        metrics = dict(raw_result.metrics)
        build_result = _build_episode_records(
            raw_result.trajectories, batch.non_tensor_batch["uid"]
        )
        metrics["compression/dropped_groups"] = len(build_result.dropped_group_uids)
        if build_result.skip_batch:
            return None, metrics

        episodes = build_result.episodes
        rewards = torch.tensor([episode.reward for episode in episodes], dtype=torch.float32).unsqueeze(-1)
        selection = _compute_rejection_sampling_selection(
            np.array([episode.group_uid for episode in episodes], dtype=object), rewards
        )
        metrics.update(
            {
                "batch/solve_none": selection.solve_none,
                "batch/solve_all": selection.solve_all,
                "batch/solve_partial": selection.solve_partial,
            }
        )
        if self.config.trainer.rejection_sample:
            episodes = [
                episode
                for episode, keep in zip(episodes, selection.valid_mask.tolist())
                if keep
            ]
            if not episodes:
                return None, metrics

        episode_batch = _build_episode_advantage_batch(episodes, self.tokenizer.pad_token_id)
        episode_batch = compute_advantage(
            episode_batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            mask_truncated_samples=self.config.algorithm.mask_truncated_samples,
            clip_advantages=self.config.algorithm.clip_advantages,
        )
        response_mask = compute_response_mask(episode_batch)
        _extract_episode_scalar_advantages(
            episodes,
            episode_batch.batch["advantages"],
            response_mask,
            episode_batch.non_tensor_batch["source_batch_idx"],
        )

        buckets = _pack_episode_optimizer_steps(episodes, episode_mini_batch_size)
        optimizer_step_by_source = {
            episode.source_batch_idx: step_id
            for step_id, bucket in enumerate(buckets)
            for episode in bucket
        }
        assignment = None
        if effective_dp_degree == 1:
            segment_batch = _transform_policy_segments(
                episodes,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                pad_token_id=self.tokenizer.pad_token_id,
                optimizer_step_by_source=optimizer_step_by_source,
                effective_dp_degree=effective_dp_degree,
            )
        else:
            assignment = _assign_episode_buckets_to_dp(
                buckets, effective_dp_degree
            )
            segment_batch = _transform_policy_segments_d2(
                episodes,
                optimizer_step_by_source=optimizer_step_by_source,
                assignment=assignment,
                effective_dp_degree=effective_dp_degree,
                sp_size=self.config.actor_rollout_ref.actor.ulysses_sequence_parallel_size,
                eos_token_id=self.tokenizer.eos_token_id,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        segment_batch.meta_info.update(
            {
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
                "multi_turn": False,
                "global_token_num": torch.sum(
                    segment_batch.batch["attention_mask"], dim=-1
                ).tolist(),
            }
        )

        # Freeze all behavior-policy/reference log probabilities before the first update.
        with _timer("old_log_prob", timing_raw):
            if effective_dp_degree == 1:
                old_log_prob = _select_unique_sp_result(
                    self.actor_rollout_wg.compute_compression_log_prob_d1(segment_batch),
                    operation="compute_compression_log_prob_d1",
                )
            else:
                old_log_prob = self.actor_rollout_wg.compute_compression_log_prob_d2(
                    segment_batch
                )
            segment_batch = segment_batch.union(old_log_prob)
        if self.use_reference_policy:
            with _timer("ref", timing_raw):
                if effective_dp_degree == 1:
                    ref_log_prob = _select_unique_sp_result(
                        self.ref_policy_wg.compute_compression_ref_log_prob_d1(segment_batch),
                        operation="compute_compression_ref_log_prob_d1",
                    )
                else:
                    ref_log_prob = self.ref_policy_wg.compute_compression_ref_log_prob_d2(
                        segment_batch
                    )
                segment_batch = segment_batch.union(ref_log_prob)

        if self.config.trainer.critic_warmup <= self.global_steps:
            with _timer("update_actor", timing_raw):
                if effective_dp_degree == 1:
                    actor_results = self.actor_rollout_wg.update_compression_actor_d1(
                        segment_batch
                    )
                    actor_output = _select_unique_sp_result(
                        actor_results, operation="update_compression_actor_d1"
                    )
                else:
                    actor_output = self.actor_rollout_wg.update_compression_actor_d2(
                        segment_batch
                    )
            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

        episode_counts = [len(bucket) for bucket in buckets]
        metrics.update(
            {
                "compression/optimizer_steps": len(buckets) * self.config.actor_rollout_ref.actor.ppo_epochs,
                "compression/episodes_per_step_min": min(episode_counts),
                "compression/episodes_per_step_max": max(episode_counts),
                "compression/episodes_per_step_mean": float(np.mean(episode_counts)),
                "compression/episodes": len(episodes),
                "compression/segments": sum(len(episode.policy_segments) for episode in episodes),
                "compression/policy_tokens": int(segment_batch.batch["policy_mask"].sum().item()),
                "compression/effective_dp_degree": effective_dp_degree,
                "compression/logical_segment_rows": len(segment_batch),
                "compression/real_segment_rows": int(segment_batch.meta_info.get("real_segment_rows", len(segment_batch))),
                "compression/dummy_rows": int(segment_batch.meta_info.get("dummy_rows", 0)),
                "compression/dummy_tokens": int(segment_batch.meta_info.get("dummy_tokens", 0)),
                "compression/physical_rows_with_dummy": len(segment_batch),
            }
        )
        if assignment is not None:
            all_dp_costs = [
                cost
                for costs in assignment.dp_costs_by_step.values()
                for cost in costs
            ]
            metrics.update(
                {
                    "compression/max_dp_cost": max(all_dp_costs),
                    "compression/min_dp_cost": min(all_dp_costs),
                    "compression/dp_cost_imbalance": max(all_dp_costs) - min(all_dp_costs),
                }
            )
        return segment_batch, metrics

    def init_workers(self):
        super().init_workers()

        # Initialize additional agent class
        # Number of agents is set to be 0 initially
        if self.hybrid_engine:
            agent_rollout_wg = self.actor_rollout_wg
        else:
            agent_rollout_wg = self.rollout_wg

        if self.config.actor_rollout_ref.rollout.mode == "async":
            rollout_engine = self.async_rollout_manager
        else:
            rollout_engine = agent_rollout_wg

        self.agent_execution_engine = AsyncAgentExecutionEngine(
            rollout_engine=rollout_engine,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=self.config.agent.use_stepwise_advantage,
            trajectory_timeout=self.config.agent.trajectory_timeout,
            overlong_filter=self.config.agent.overlong_filter,
            **self.config.agent.get("engine_args", {}),
        )

    def init_envs_and_agents(self, batch):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        env_args = batch.non_tensor_batch["extra_info"].tolist()
        # pprint(f"!!!!!! look fit_agent init_envs_and_agents env_args length:{len(env_args)},\n detailed info:{env_args}")
        pprint(f"!!!!!! look fit_agent init_envs_and_agents env_args length:{len(env_args)},\n")

        full_agent_args = dict(self.config.agent.get("agent_args", {})) | self.agent_args
        base_env_args = dict(self.config.env.get("env_args", {})) | self.env_args

        def _create_env(i):
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            return i, self.env_class.from_dict({**env_args[i], **base_env_args})

        def _create_agent(i):
            return i, self.agent_class(**full_agent_args)

        # Create environments in parallel while preserving order
        envs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=64) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(len(env_args))]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env

        # Create agents in parallel while preserving order
        agents = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=64) as executor:
            agent_futures = [executor.submit(_create_agent, i) for i in range(len(envs))]
            for future in as_completed(agent_futures):
                idx, agent = future.result()
                agents[idx] = agent
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        print(f"!!!!!!!!!!!!!!! now in fit_agent")
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        compression_enabled = bool(
            self.agent_args.get("enable_context_compression", False)
        )
        compression_effective_dp_degree = None
        compression_episode_mini_batch_size = None
        if compression_enabled:
            compression_effective_dp_degree = (
                self._query_compression_effective_dp_degree()
            )
            compression_episode_mini_batch_size = (
                self._validate_compression_config(
                    compression_effective_dp_degree
                )
            )

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"!!!!!! now in fit_agent, epoch {epoch}")
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                print(f"now in fit_agent, batch, epoch {epoch}, step {self.global_steps}")
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                metrics = {}
                timing_raw = {}

                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                batch.meta_info = {
                    "agent_rollout": True,  # no need to generate multiple ones since environment is repeated already
                }

                if compression_enabled:
                    with _timer("step", timing_raw):
                        pprint(
                            f"!!!!!!!!step {self.global_steps} compression D{compression_effective_dp_degree}: initializing envs and agents"
                        )
                        self.init_envs_and_agents(batch)
                        compression_batch, compression_metrics = (
                            self._run_compression_training_step(
                                batch=batch,
                                timing_raw=timing_raw,
                                effective_dp_degree=compression_effective_dp_degree,
                                episode_mini_batch_size=compression_episode_mini_batch_size,
                            )
                        )
                        metrics.update(compression_metrics)
                        if compression_batch is None:
                            continue

                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.test_freq > 0
                            and self.global_steps % self.config.trainer.test_freq == 0
                        ):
                            with _timer("testing", timing_raw):
                                metrics.update(self._validate_agent())
                        if (
                            self.config.trainer.save_freq > 0
                            and self.global_steps % self.config.trainer.save_freq == 0
                        ):
                            with _timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()

                    metrics.update(
                        compute_timing_metrics(
                            batch=compression_batch, timing_raw=timing_raw
                        )
                    )
                    logger.log(data=metrics, step=self.global_steps)
                    self.global_steps += 1
                    if self.global_steps >= self.total_training_steps:
                        if self.val_reward_fn is not None:
                            val_metrics = self._validate_agent()
                            logger.log(data=val_metrics, step=self.global_steps)
                        return
                    continue

                with _timer("step", timing_raw):
                    pprint(f"!!!!!!!!step {self.global_steps} started, now initializing envs and agents")
                    self.init_envs_and_agents(batch)
                    pprint(f"!!!!!!!!step {self.global_steps} started, successfully initialized envs and agents")

                    if self.config.agent.use_stepwise_advantage:
                        pprint(f"!!!!!!!! use stepwise advantage")
                        final_gen_batch_output = self.generate_agent_steps(timing_raw=timing_raw, meta_info=batch.meta_info, uids=batch.non_tensor_batch["uid"])
                        repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                        # need to repeat to make shape match
                        batch = batch.repeat_by_counts(repeat_counts, interleave=True)
                        final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                        # batch needs to be padded to divisor of world size, we will pad with everything masked out
                        batch = batch.union(final_gen_batch_output)
                        batch = self._pad_dataproto_to_world_size(batch=batch)
                    else:
                        pprint(f"!!!!!!!! use trajectory")
                        pprint(f"!!!!!!!! batch meta info:{batch.meta_info}")
                        final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch.meta_info)
                        batch = batch.union(final_gen_batch_output)
                        metrics.update(generate_metrics)
                        print(f"!!!!!! finsh trajectory")

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # reward tensor for env-based trajectory data can be obtained by processing the trajectories
                        if "token_level_scores" not in batch.batch:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"]  # filled in by environment collected trajectory transformation

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                            # Check if all rewards are <= 0 or all are 1 >= for this uid
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards >= 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                        # Log to metrics
                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all
                        metrics["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all

                        if self.config.trainer.rejection_sample:
                            # log the actual complete training rewards before rejection sampling
                            token_level_rewards = None  # for metrics calculation
                            if self.config.agent.use_stepwise_advantage:
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                non_pad_steps = batch.select_idxs(non_pad_step_indices)
                                is_last_step = non_pad_steps.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)
                                token_level_rewards = last_step_batch.batch["token_level_scores"]
                            else:
                                token_level_rewards = batch.batch["token_level_scores"]
                            full_sequence_score = token_level_rewards.sum(-1)
                            metrics["critic/full-score/mean"] = torch.mean(full_sequence_score).detach().item()
                            metrics["critic/full-score/max"] = torch.max(full_sequence_score).detach().item()
                            metrics["critic/full-score/min"] = torch.min(full_sequence_score).detach().item()

                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]

                            if self.config.agent.use_stepwise_advantage and self.config.agent.stepwise_advantage_mode == "broadcast":
                                # batch now only contains steps with valid uids
                                # filter out padding steps
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps

                                # need to make sure both number of last steps (number of uids) and number of total steps in the batch (batch size after processing) are all multiples of world size
                                # separate out last step and intermediate steps
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                not_last_step_indices = np.where(is_last_step == False)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                                non_last_step_batch = batch.select_idxs(not_last_step_indices)

                                # filter last_step_batch to make sure its multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (
                                    last_step_batch.batch["input_ids"].shape[0]  # 1 per trajectory
                                    // num_trainer_replicas
                                ) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                last_step_batch = last_step_batch[size_mask]  # filtered last steps

                                # now we go through all the non_last_step_batch and keep everything that has same idxs that exists in the filtered last steps
                                valid_last_step_idxs = last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_idxs = non_last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_mask = np.isin(non_last_step_idxs, valid_last_step_idxs)
                                non_last_step_batch = non_last_step_batch[non_last_step_mask]

                                # concatenate then pad
                                batch = DataProto.concat([last_step_batch, non_last_step_batch])
                                batch = self._pad_dataproto_to_world_size(batch)
                            else:
                                # Round down to the nearest multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                batch = batch[size_mask]

                        # recompute old_log_probs
                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.agent.use_stepwise_advantage:
                            if self.config.agent.stepwise_advantage_mode == "mc_return":
                                batch.batch["token_level_rewards"] = batch.batch["mc_returns"]
                                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]

                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
                            elif self.config.agent.stepwise_advantage_mode == "broadcast":
                                # In case of step-wise advantage broadcast, we would split out the final steps, then merge again
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                last_step_indices = np.where(is_last_step == True)[0]
                                other_step_indices = np.where(is_last_step == False)[0]
                                other_step_batch = batch.select_idxs(other_step_indices)
                                batch = batch.select_idxs(last_step_indices)  # This batch only has last steps
                            else:
                                raise ValueError(f"Stepwise advantage mode {self.config.agent.stepwise_advantage_mode} not supported")

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            mask_truncated_samples=self.config.algorithm.mask_truncated_samples,
                            clip_advantages=self.config.algorithm.clip_advantages,
                        )

                        if self.config.agent.use_stepwise_advantage and self.config.agent.stepwise_advantage_mode == "broadcast":
                            # remove the padded last steps
                            # Merging the separated out steps using the advantage from last steps
                            self._stepwise_advantage_broadcast(batch, other_step_batch=other_step_batch)
                            # batch = batch.merge(other_step_batch)
                            batch = DataProto.concat([batch, other_step_batch])

                    batch = self._pad_dataproto_to_world_size(batch=batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                pprint(f"!!!!!!! now logger metrics ")
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _validate_agent(self):
        rewards_lst = []
        data_source_lst = []
        uid_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
                "agent_rollout": True,
            }
            self.init_envs_and_agents(test_batch)

            if self.config.agent.use_stepwise_advantage:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
                # for validation, we only need the last step
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)  # This batch only has last steps
            else:
                test_output_gen_batch, _ = self.generate_agent_trajectory(meta_info=test_batch.meta_info)

            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor = test_batch.batch["token_level_scores"]

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            uid_lst.append(test_batch.non_tensor_batch["uid"])

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}

        # to group for pass@k
        uid_tensor = np.concatenate(uid_lst, axis=0)
        data_source_uid_pass_rates = {}  # data source to {uid: pass or not}

        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            # pass@k
            if data_source not in data_source_uid_pass_rates:
                data_source_uid_pass_rates[data_source] = {}

            uid = uid_tensor[i]
            if uid not in data_source_uid_pass_rates[data_source]:
                data_source_uid_pass_rates[data_source][uid] = 0  # default to not pass
            # take highest score
            data_source_uid_pass_rates[data_source][uid] = max(data_source_uid_pass_rates[data_source][uid], reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            # clip rewards to be between 0 and 1
            rewards_array = np.array(rewards)
            rewards_array = np.clip(rewards_array, 0, 1)
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards_array)

        for data_source, pass_rates in data_source_uid_pass_rates.items():
            pass_k_lst = []
            for uid, pass_score in pass_rates.items():
                pass_k_lst.append(pass_score >= 1)  # assuming 1 means passed
            metric_dict[f"val/test_score/pass@k/{data_source}"] = np.mean(pass_k_lst)

        return metric_dict

    def generate_agent_trajectory(
        self, timing_raw=None, meta_info=None, return_raw_trajectories=False
    ):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards

        Args:
            envs: The environments in which the agent interacts.
            agents: The agents to use for interation.
            timing_raw: Dictionary to store timing information for profiling.
            meta_info (optional): Metadata for veRL generation.

        Returns:
            DataProto: Representation of the agent's trajectories.
            Dict[str:float]: Metrics for the generation process.
        """
        if timing_raw is None:
            timing_raw = {}
        with _timer("collect_trajectory", timing_raw):
            trajectories = []
            if self.config.agent.async_engine:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for _, trajectory in enumerate(gen_seq_generator):
                    trajectories.append(trajectory)
            else:
                # generate_trajectories returns list of trajectories.
                trajectories = self.agent_execution_engine.generate_trajectories(timing_raw=timing_raw, mode="Token", meta_info=meta_info)
        # Sort trajectories by their idx, to ensure they are in order.
        trajectories.sort(key=lambda x: x["idx"])

        with _timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories)
        if return_raw_trajectories:
            return AgentTrajectoryGenerationResult(
                trajectories=trajectories,
                transformed=final_gen_batch_output,
                metrics=metrics,
            )
        return final_gen_batch_output, metrics

    def generate_agent_steps(self, timing_raw=None, meta_info=None, uids=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards.

        Returns:
            DataProto: Representation of the last step of agent's trajectories.
            Dict[str:List[DataProto]]: Index of the trajectory to the rest of the steps from the trajectory.
        """
        if timing_raw is None:
            timing_raw = {}
        if uids is None:
            uids = []
        with _timer("collect_trajectory", timing_raw):
            steps = []
            if self.config.agent.async_engine:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Step")
                for _, trajectory in enumerate(gen_seq_generator):
                    steps.append(trajectory)
            else:
                # generate_trajectories returns list of trajectories.
                steps = self.agent_execution_engine.generate_trajectories(timing_raw=timing_raw, mode="Step", meta_info=meta_info)
        # Sort trajectories by their idx, to ensure they are in order.
        steps.sort(key=lambda x: x["idx"])

        with _timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output = self._transform_agent_steps(steps, uids=uids)
        return final_gen_batch_output

    def _transform_agent_trajectories(self, trajectories: list[dict]):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """
        from verl.utils.torch_functional import pad_sequence_to_length

        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_masks_list = []
        traj_scores = []
        chat_completions = []
        traj_metrics = []
        termination_reasons = []
        # timeout_infos = []
        traj_metrics_ori = []
        metrics = {}
        empyt_num = 0
        for traj in trajectories:
            prompt_tokens = traj["prompt_tokens"]
            response_tokens = traj["response_tokens"]
            # test if trajectory is empty
            # assert prompt_tokens.numel() != 0 and response_tokens.numel() != 0, f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} of trajectory shouldn't be empty. Please check make sure environment is working and the config"
            if prompt_tokens.numel() == 0 or response_tokens.numel() == 0:
                print(f"Ops!!!!!Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} of trajectory shouldn't be empty. Please check make sure environment is working and the config")
                empyt_num+=1
                print(f"Ops!!! Now empty num is {empyt_num}")
                continue
            

            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_masks_list.append(traj["response_masks"])
            traj_scores.append(traj["trajectory_reward"])
            termination_reasons.append(traj["termination_reason"])
            # timeout_infos.append(traj["timeout_info"])
            chat_completions.append(traj["chat_completions"])
            traj_metrics.append(traj["metrics"])
            traj_metrics_ori.append(traj["metrics"])
        if empyt_num == len(trajectories):
            print(f"Ops!!! All failed! empty num is {empyt_num}")

        # Flatten traj_metrics into a dict of lists
        traj_metrics = {k: [d[k] for d in traj_metrics] for k in traj_metrics[0]}
        # Aggregate metrics (mean, min, max)
        for k, v_list in traj_metrics.items():
            v_list = [v for v in v_list if v is not None and v >= 0]
            if not v_list:
                continue
            v_list = np.array(v_list)
            metrics.update(
                {
                    f"traj/{k}_mean": v_list.mean(),
                    f"traj/{k}_min": v_list.min(),
                    f"traj/{k}_max": v_list.max(),
                }
            )

        # Save chat completions to a file
        save_dir = os.path.join(self.config.trainer.default_local_dir, "chat_completions")
        os.makedirs(save_dir, exist_ok=True)
        # Save it into a jsonl files (self.global_steps)
        with open(os.path.join(save_dir, f"{self.global_steps}.jsonl"), "w") as f:
            # if len(chat_completions)==len(traj_scores)==len(traj_metrics_ori)==len(agent_patches):
            if len(chat_completions)==len(traj_scores):
                print(f"Success!!! Data is aligned for this batch! Current batch: {self.global_steps}\nlen(chat_completions):{len(chat_completions)};")
                sample_num = len(chat_completions)
                for idx_write in range(sample_num):
                    try:
                        obj_now = {
                            "chat_completion":chat_completions[idx_write],
                            "rendered_prompt": trajectories[idx_write].get("rendered_prompt", ""),
                            "chat_template_parser": trajectories[idx_write].get("chat_template_parser", ""),
                            "prompt_token_count": int(trajectories[idx_write]["prompt_tokens"].numel()),
                            "traj_score":float(traj_scores[idx_write]),
                            "traj_metric":traj_metrics_ori[idx_write],
                            "pre_training_rollout": bool(self.global_steps == 1),
                            "compression_events": trajectories[idx_write].get("compression_events", []),
                            "forced_compression_candidates": trajectories[idx_write].get("forced_compression_candidates", []),
                            # "global_step":int(self.global_steps),
                            "termination_reason":termination_reasons[idx_write],
                            # "timeout_info":timeout_infos[idx_write]
                        }
                        f.write(json.dumps(obj_now, ensure_ascii=False) + "\n")
                    except:
                        
                        # f.write(agent_patches[idx_write] + "\n")
                        obj_now = {
                            "chat_completion":chat_completions[idx_write],
                        }
                        f.write(json.dumps(obj_now, ensure_ascii=False) + "\n")
                        # print(f"Failed to write trajectory!!!!:\n {agent_patches[idx_write]}")
            else:
                print(f"Ops!!! Data is misaligned for this batch! Current batch: {self.global_steps}\nlen(chat_completions):{len(chat_completions)};len(traj_scores):{len(traj_scores)};len(traj_metrics):{len(traj_metrics)}")
                for chat_completion in chat_completions:
                    f.write(json.dumps(chat_completion) + "\n")

        # reverse the list and create tensors, pad, then flip to achieve left padding
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])

        prompts_batch = pad_sequence_to_length(prompts_batch, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)

        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)

        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)

        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)

        attention_mask = torch.where(trajectory_batch != self.tokenizer.pad_token_id, 1, 0)

        # Compute position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        for i, traj_score in enumerate(traj_scores):
            last_valid_idx = valid_response_length_sequences[i] - 1
            if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                score_batch[i, last_valid_idx] = traj_score

        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "traj_mask": traj_mask,
        }

        self.visualize_trajectory(DataProto.from_dict(tensors=tensor_batch))

        return DataProto.from_dict(tensors=tensor_batch), metrics

    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="traj_mask"):
        """
        Visualize the trajectory from tensor_batch by detokenizing prompts and responses,
        and highlighting the masked parts with color.

        Args:
            tensor_batch: The tensor batch containing trajectory data
            sample_idx: Starting index of samples to visualize
            max_samples: Maximum number of samples to visualize
        """
        from rllm.misc import colorful_print

        # Get the relevant tensors
        prompts = tensor_batch.batch["prompts"]
        responses = tensor_batch.batch["responses"]
        traj_mask = tensor_batch.batch[mask_key]
        token_level_scores = tensor_batch.batch["token_level_scores"]

        batch_size = prompts.shape[0]
        end_idx = min(sample_idx + max_samples, batch_size)

        for i in range(sample_idx, end_idx):
            # colorful_print(f"\n===== Sample {i} =====", fg="cyan", bold=True)

            # Detokenize prompt
            prompt_tokens = prompts[i]
            prompt_mask = prompt_tokens != self.tokenizer.pad_token_id
            valid_prompt_tokens = prompt_tokens[prompt_mask]
            prompt_text = self.tokenizer.decode(valid_prompt_tokens)

            # colorful_print("Prompt:", fg="green", bold=True)
            # colorful_print(f"{prompt_text}\n", fg="green")

            # Detokenize response with color highlighting for masked tokens
            response_tokens = responses[i]
            response_mask = traj_mask[i]

            # Get non-padding tokens
            valid_indices = response_tokens != self.tokenizer.pad_token_id
            valid_response_tokens = response_tokens[valid_indices]
            valid_response_mask = response_mask[valid_indices]

            # Then show token-by-token with masking
            # colorful_print("Response with masking:", fg="yellow", bold=True)

            for j, (token, mask) in enumerate(zip(valid_response_tokens, valid_response_mask, strict=False)):
                token_text = self.tokenizer.decode(token)

                # Check if this token has a reward
                has_reward = token_level_scores[i, j] != 0

                # Apply different colors based on mask and rewards
                if mask == 0:
                    # Masked token (not used in training)
                    pass
                    # print(token_text)
                    # colorful_print(token_text, fg="red", end="")
                elif has_reward:
                    # Token with reward
                    # print(token_text)
                    # colorful_print(token_text, bg="green", end="")
                    reward_info = ""
                    if has_reward:
                        reward_info += f" R:{token_level_scores[i, j].item():.2f}"
                    
                    # print(reward_info)
                    # colorful_print(reward_info, fg="magenta", end="")
                else:
                    pass
                    # Normal token used in training
                    # colorful_print(token_text, fg="blue", end="")
                    # print(token_text)
            print()  # New line after all tokens

            # Print reward summary
            total_reward = token_level_scores[i].sum().item()
            # colorful_print("Rewards:", fg="green", bold=True)
            print(f" Trajectory Reward={total_reward:.2f}")

    def generate_agent_trajectories_async(self, timing_raw=None, meta_info=None, mode="Token"):
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        This method runs the asynchronous `trajectory_generator` in a
        separate thread and yields the results synchronously through a queue.
        This allows the main training loop (which might be synchronous) to consume
        asynchronously generated trajectories.

        Args:
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to {}.
            meta_info (dict, optional): Additional metadata for the generation process. Defaults to None.

        Yields:
            Any: Items generated by the `trajectory_generator`, typically
                 representing parts or results of agent trajectories in token format.
        """
        if timing_raw is None:
            timing_raw = {}
        queue = Queue()

        def runner():
            async def consume():
                async for item in self.agent_execution_engine.trajectory_generator(timing_raw=timing_raw, mode=mode, meta_info=meta_info):
                    queue.put(item)
                queue.put(None)  # sentinel to signal done

            asyncio.run(consume())

        Thread(target=runner, daemon=True).start()
        while True:
            item = queue.get()
            if item is None:
                break
            yield item

    def _transform_agent_steps(self, steps: list[dict], uids: np.ndarray):
        from verl.utils.torch_functional import pad_sequence_to_length

        all_prompts_list = []
        all_responses_list = []

        step_numbers = []  # number of steps of each episode, 0 indexed
        all_steps_idx_list = []
        all_steps_is_last_step_list = []
        all_steps_step_num = []  # total number of steps the trajectory this step belongs to have
        all_steps_step_ids = []
        training_rewards = []
        all_mc_returns = []  # Monte Carlo returns for each episode
        # the last step will have reward assigned and be used for advantage calculation

        for episode in steps:
            episode_steps = episode["steps"]
            idx = episode["idx"]
            training_reward = episode["trajectory_reward"]
            mc_returns = episode["mc_returns"]

            all_prompts_list.extend([torch.tensor(self.tokenizer.encode(s["prompt"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])
            all_responses_list.extend([torch.tensor(self.tokenizer.encode(s["response"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])

            step_numbers.append(len(episode_steps) - 1)
            training_rewards.append(training_reward)
            all_mc_returns.extend(mc_returns)

            all_steps_idx_list.extend([idx for _ in range(len(episode_steps))])
            all_steps_is_last_step_list.extend([False for _ in range(len(episode_steps))])
            all_steps_is_last_step_list[-1] = True

            all_steps_step_num.extend([len(episode_steps) for _ in range(len(episode_steps))])
            all_steps_step_ids.extend([f"{uids[idx]}_step{i}" for i in range(len(episode_steps))])

        # Convert all steps into token tensors
        # reverse the list and create tensors, pad, then flip to achieve left padding
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_prompts_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])

        prompts_batch = pad_sequence_to_length(prompts_batch, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)

        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_responses_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)

        complete_step_batch = torch.concat([prompts_batch, response_batch], dim=1)
        attention_mask = torch.where(complete_step_batch != self.tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # same as regular repsonse_mask, padded tensors will have this zeroed out
        traj_mask = torch.where(response_batch != self.tokenizer.pad_token_id, 1, 0)

        # Place all rewards to last response token of the last_step response
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        mc_return_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        # reward is given for last token of every step for logging purposes, but only last steps will be used to calculate advantage
        step_index = 0
        for i, traj_score in enumerate(training_rewards):
            step_num = step_numbers[i] + 1  # since step_numbers is 0 indexed
            for _ in range(step_num):
                last_valid_idx = valid_response_length_sequences[step_index] - 1
                if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                    score_batch[step_index, last_valid_idx] = traj_score
                    mc_return_batch[step_index, last_valid_idx] = all_mc_returns[step_index]
                step_index += 1
        assert step_index == score_batch.shape[0], f"Number of total steps used should equal to batch size, but got {step_index} and {score_batch.shape[0]}"

        tensor_batch = {
            "input_ids": complete_step_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "mc_returns": mc_return_batch,
            "traj_mask": traj_mask,
        }

        batch_id = str(uuid.uuid4())
        non_tensor_batch = {
            "idxs": np.array(all_steps_idx_list),
            "step_nums": np.array(all_steps_step_num),
            "is_last_step": np.array(all_steps_is_last_step_list),
            "is_pad_step": np.array([False for _ in range(len(all_steps_idx_list))]),
            "batch_id": np.array([batch_id for _ in range(len(all_steps_idx_list))]),  # in case need to differentiate which iteration the step is coming from
            "step_ids": np.array(all_steps_step_ids),
        }

        meta_info = {"repeat_counts": [x + 1 for x in step_numbers]}

        result = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)

        # Find indices of last steps for visualization
        last_step_indices = [i for i, is_last in enumerate(non_tensor_batch["is_last_step"]) if is_last]
        if last_step_indices:
            sample_indices = np.random.choice(last_step_indices, size=min(2, len(last_step_indices)), replace=False)
            for idx in sample_indices:
                self.visualize_trajectory(result, sample_idx=idx, max_samples=1)
        return result

    def _stepwise_advantage_broadcast(self, last_step_batch, other_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants
        if "response_mask" not in other_step_batch.batch.keys():
            other_step_batch.batch["response_mask"] = compute_response_mask(other_step_batch)
        if "response_mask" not in last_step_batch.batch.keys():
            last_step_batch.batch["response_mask"] = compute_response_mask(last_step_batch)
        src_indices = last_step_batch.non_tensor_batch["idxs"]
        src_total_steps = last_step_batch.non_tensor_batch["step_nums"]
        tgt_indices = other_step_batch.non_tensor_batch["idxs"]
        src_advantages = last_step_batch.batch["advantages"]
        src_mask = last_step_batch.batch["response_mask"]
        tgt_mask = other_step_batch.batch["response_mask"]

        # Build idx -> scalar advantage
        idx_to_scalar_adv = {}
        for i, idx in enumerate(src_indices):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.agent.normalize_step_advantage:
                # normalize the advantage against number of steps
                scalar = scalar / src_total_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            idx_to_scalar_adv[int(idx)] = scalar

        # Create new tensor for other_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=idx_to_scalar_adv[int(idx)], dtype=torch.float32) for i, idx in enumerate(tgt_indices)])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        other_step_batch.batch["advantages"] = final_advantage
        other_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            batch.non_tensor_batch["is_last_step"][idx] = False
            batch.non_tensor_batch["is_pad_step"][idx] = True

        return batch
