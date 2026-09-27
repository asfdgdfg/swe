"""Integration-only Stage 9C-2: real rollout through one compression PPO update."""

import asyncio
import inspect
import json
import os
from pathlib import Path

import hydra
import numpy as np
import ray


REPORT = Path("/tmp/stage9c2_report.json")
RESPONSES = Path("/tmp/stage9c2_forced_responses.jsonl")
SUMMARY = (
    "Continue solving the original repository task. The repository has been inspected "
    "and an initial tool action was executed. Preserve all task requirements, inspect "
    "the relevant source and tests, implement the fix, and verify it."
)
FORCED_RESPONSE = (
    "<function=compress_context>\n<parameter=summary>\n"
    + SUMMARY
    + "\n</parameter>\n</function>"
)


def write_report(update):
    current = {}
    if REPORT.exists():
        current = json.loads(REPORT.read_text(encoding="utf-8"))
    current.update(update)
    REPORT.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


@hydra.main(
    config_path="../../rllm/trainer/config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config):
    REPORT.unlink(missing_ok=True)
    RESPONSES.unlink(missing_ok=True)
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "working_dir": "./DeepSWE_RL/rllm/verl",
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "PYTHONPATH": "./DeepSWE_RL/rllm",
                },
            }
        )
    ray.get(run_stage9c2.remote(config))


@ray.remote(num_cpus=1)
def run_stage9c2(config):
    import sys
    sys.path.insert(0, "./R2E-Gym/src")
    sys.path.insert(0, "./DeepSWE_RL/rllm/verl")

    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, CriticWorker

    import rllm.trainer.verl.agent_ppo_trainer as trainer_module
    from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
    from rllm.agents.swe_agent import SWEAgent, parse_xml_response
    from rllm.environments.swe.swe_arsenal import SWEEnv

    OmegaConf.resolve(config)
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))

    class ControlledRewardSWEEnv(SWEEnv):
        @staticmethod
        def from_dict(extra_info):
            if isinstance(extra_info, str):
                extra_info = json.loads(extra_info)
            signature = inspect.signature(SWEEnv.__init__)
            kwargs = {
                name: extra_info[name]
                for name in signature.parameters
                if name != "self" and name in extra_info
            }
            kwargs["entry"] = extra_info
            return ControlledRewardSWEEnv(**kwargs)

        def compute_final_reward(self):
            # Engine halves MAX_STEPS rewards. This yields final episode rewards 1/0.
            return 2.0 if int(self.idx) % 2 == 0 else 0.0

    class ForcedCompressionEngine(AsyncAgentExecutionEngine):
        async def get_model_response(self, prompt, application_id, **kwargs):
            if not hasattr(self, "_stage9c_calls"):
                self._stage9c_calls = {}
                self._stage9c_target = None
            if self._stage9c_target is None:
                self._stage9c_target = application_id
            call_index = self._stage9c_calls.get(application_id, 0)
            self._stage9c_calls[application_id] = call_index + 1
            forced = application_id == self._stage9c_target and call_index == 1
            if forced:
                response = FORCED_RESPONSE
            else:
                response = await super().get_model_response(prompt, application_id, **kwargs)
            record = {
                "application_id": application_id,
                "call_index": call_index,
                "forced": forced,
                "response": response,
            }
            try:
                _, action = parse_xml_response(response)
                record["function_name"] = action.function_name
                record["parameters"] = action.parameters
            except Exception as error:
                record["parse_error"] = repr(error)
            with RESPONSES.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            return response

    trainer_module.AsyncAgentExecutionEngine = ForcedCompressionEngine

    diagnostics = {}
    original_build = trainer_module._build_episode_records
    def recording_build(trajectories, group_uids):
        result = original_build(trajectories, group_uids)
        diagnostics["episodes"] = [
            {
                "source_batch_idx": episode.source_batch_idx,
                "episode_id": episode.episode_id,
                "group_uid": episode.group_uid,
                "reward": episode.reward,
                "termination_reason": episode.termination_reason,
                "segments": len(episode.policy_segments),
                "policy_tokens": sum(sum(segment["policy_mask"]) for segment in episode.policy_segments),
            }
            for episode in result.episodes
        ]
        write_report({"episodes": diagnostics["episodes"]})
        return result
    trainer_module._build_episode_records = recording_build

    original_extract = trainer_module._extract_episode_scalar_advantages
    def recording_extract(episodes, *args, **kwargs):
        result = original_extract(episodes, *args, **kwargs)
        values = {episode.episode_id: episode.scalar_advantage for episode in episodes}
        diagnostics["advantages"] = values
        write_report({"advantages": values})
        return result
    trainer_module._extract_episode_scalar_advantages = recording_extract

    original_assign = trainer_module._assign_episode_buckets_to_dp
    def recording_assign(buckets, effective_dp_degree):
        result = original_assign(buckets, effective_dp_degree)
        payload = {
            "target_dp_by_source": result.target_dp_by_source,
            "episode_cost_by_source": result.episode_cost_by_source,
            "dp_costs_by_step": result.dp_costs_by_step,
        }
        diagnostics["assignment"] = payload
        write_report({"assignment": payload})
        return result
    trainer_module._assign_episode_buckets_to_dp = recording_assign

    original_transform = trainer_module._transform_policy_segments_d2
    def recording_transform(episodes, **kwargs):
        batch = original_transform(episodes, **kwargs)
        real = ~batch.non_tensor_batch["is_dummy"]
        dummy = batch.non_tensor_batch["is_dummy"]
        episode_weight_sums = {}
        for row in np.flatnonzero(real):
            episode_id = str(batch.non_tensor_batch["episode_id"][row])
            episode_weight_sums[episode_id] = episode_weight_sums.get(episode_id, 0.0) + float(
                batch.batch["episode_normalized_weight"][row].sum().item()
            )
        payload = {
            "temperature": float(config.actor_rollout_ref.rollout.temperature),
            "real_rows": int(real.sum()),
            "dummy_rows": int(dummy.sum()),
            "real_actor_weight_sum": float(batch.batch["actor_token_weight"][real].sum().item()),
            "dummy_actor_weight_sum": float(batch.batch["actor_token_weight"][dummy].sum().item()),
            "episode_weight_sums": episode_weight_sums,
            "optimizer_step_ids": [int(x) for x in batch.non_tensor_batch["optimizer_step_id"]],
            "target_dp_replicas": [int(x) for x in batch.non_tensor_batch["target_dp_replica"]],
            "meta_info": {key: value for key, value in batch.meta_info.items() if isinstance(value, (int, float, str, bool))},
        }
        diagnostics["transform"] = payload
        write_report({"transform": payload})
        return batch
    trainer_module._transform_policy_segments_d2 = recording_transform

    actor_cls = ray.remote(max_concurrency=2048)(AsyncActorRolloutRefWorker)
    critic_cls = ray.remote(CriticWorker)
    role_worker_mapping = {Role.ActorRollout: actor_cls, Role.Critic: critic_cls}
    pool = "global_pool"
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={pool: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
        mapping={Role.ActorRollout: pool, Role.Critic: pool},
    )
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0)
    trainer = trainer_module.AgentPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn,
        val_reward_fn=None,
        env_class=ControlledRewardSWEEnv,
        agent_class=SWEAgent,
        env_args=dict(config.env.get("env_args", {})),
        agent_args=dict(config.agent.get("agent_args", {})),
    )
    write_report({"status": "initializing", "controlled_reward": True, "forced_summary": SUMMARY})
    trainer.init_workers()
    write_report({"status": "training"})
    trainer.fit_agent()
    write_report({"status": "complete"})


if __name__ == "__main__":
    main()
