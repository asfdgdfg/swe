#!/usr/bin/env bash
set -euo pipefail

# SWE-smith 172-image run.  Start from the repository root because
# train_agent_ppo.py resolves Ray's working_dir relative to this directory.
cd /home/yyk/yyk11/zhongtianyang/memory/SWE-Master
PY=/home/yyk/yyk11/zhongtianyang/memory/SWE-Master/DeepSWE_RL/rllm/.venv/bin/python
test -x "$PY"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=./DeepSWE_RL/rllm/rllm:${PYTHONPATH:-}

exec "$PY" -u -m rllm.trainer.verl.train_agent_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/home/yyk/yyk11/zhongtianyang/memory/SWE-Master/DeepSWE_RL/rllm/rllm/data/datasets/SWE_SMITH_FULL_NONEMPTY_VERIFIED/train_verl.parquet \
  data.val_files=/home/yyk/yyk11/zhongtianyang/memory/SWE-Master/DeepSWE_RL/rllm/rllm/data/datasets/SWE_SMITH_FULL_NONEMPTY_VERIFIED/train_verl.parquet \
  data.max_prompt_length=8192 data.max_response_length=51200 data.train_batch_size=8 \
  actor_rollout_ref.model.path=/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=57344 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=57344 \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.chat_scheduler=verl.workers.rollout.async_server.ChatCompletionScheduler \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=57344 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
  actor_rollout_ref.rollout.max_model_len=57344 \
  actor_rollout_ref.rollout.max_num_batched_tokens=57344 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.temperature=0.3 \
  agent.name=sweagent agent.max_steps=200 agent.async_engine=True \
  +agent.agent_args.scaffold=sweagent \
  env.name=swe_arsenal +env.env_args.backend=docker \
  +env.env_args.scaffold=sweagent +env.env_args.delete_image=False \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 trainer.logger='[console]' \
  trainer.project_name=swe_master trainer.experiment_name=qwen3_4b_len51200_20260925 trainer.default_local_dir=/home/yyk/yyk11/zhongtianyang/memory/SWE-Master/DeepSWE_RL/rllm/checkpoints/swe_master/qwen3_4b_len51200_20260925 \
  trainer.resume_mode=disable trainer.val_before_train=False \
  trainer.save_freq=10 trainer.test_freq=-1 trainer.total_epochs=1
