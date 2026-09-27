#!/usr/bin/env bash
set -euo pipefail

# Formal Qwen3-4B SWE-smith compression training. This keeps the established
# Qwen3 launch configuration and enables the Stage 3-9C compression path.
ROOT=/home/yyk/yyk11/zhongtianyang/memory/SWE-Master
RLLM="$ROOT/DeepSWE_RL/rllm"
PY="$RLLM/.venv/bin/python"
EXPERIMENT=qwen3_4b_hybrid_minbudget2048_pilot_20260928

cd "$ROOT"
test -x "$PY"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=./DeepSWE_RL/rllm:./DeepSWE_RL/rllm/verl:${PYTHONPATH:-}
export WANDB_MODE=disabled

CHECKPOINT_DIR="$RLLM/checkpoints/swe_master/$EXPERIMENT"
ROLLOUT_DIR="$RLLM/rllm/train_output_rollout/swe_master/$EXPERIMENT"
mkdir -p "$CHECKPOINT_DIR" "$ROLLOUT_DIR"

exec "$PY" -u -m rllm.trainer.verl.train_agent_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$RLLM/rllm/data/datasets/SWE_SMITH_FULL_NONEMPTY_VERIFIED/train_verl.parquet" \
  data.val_files="$RLLM/rllm/data/datasets/SWE_SMITH_FULL_NONEMPTY_VERIFIED/train_verl.parquet" \
  data.max_prompt_length=8192 \
  data.max_response_length=51200 \
  data.train_batch_size=8 \
  actor_rollout_ref.model.path=/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=57344 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=57344 \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.chat_scheduler=verl.workers.rollout.async_server.ChatCompletionScheduler \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=57344 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
  actor_rollout_ref.rollout.max_model_len=57344 \
  actor_rollout_ref.rollout.max_num_batched_tokens=57344 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=0.3 \
  agent.name=sweagent \
  agent.max_steps=200 \
  agent.async_engine=True \
  agent.compression_episode_mini_batch_size=8 \
  +agent.agent_args.scaffold=sweagent \
  +agent.agent_args.enable_context_compression=True \
  +agent.agent_args.context_compression_mode=hybrid \
  +agent.agent_args.compression_warning_tokens=32768 \
  +agent.agent_args.compression_force_tokens=40960 \
  +agent.agent_args.compression_summary_max_tokens=1024 \
  +agent.agent_args.compression_retry_num=2 \
  +agent.agent_args.compression_force_min_response_budget=2048 \
  +agent.engine_args.env_response_timeout=3600 \
  +agent.engine_args.env_response_retry_num=3 \
  env.name=swe_arsenal \
  +env.env_args.backend=docker \
  +env.env_args.scaffold=sweagent \
  +env.env_args.delete_image=False \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.logger='[console]' \
  trainer.project_name=swe_master \
  trainer.experiment_name="$EXPERIMENT" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.rollout_data_dir="$ROLLOUT_DIR" \
  trainer.resume_mode=disable \
  trainer.val_before_train=False \
  trainer.save_freq=10 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1
