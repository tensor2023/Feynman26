#!/usr/bin/env bash
set -euo pipefail

TRAIN_FILES="${TRAIN_FILES:-data/feynman/train.parquet}"
VAL_FILES="${VAL_FILES:-data/feynman/val.parquet}"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-Qwen/Qwen3-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-feynman26-rlvf}"
N_GPUS="${N_GPUS:-8}"

python3 train_ppo.py \
  algorithm.adv_estimator=foldgrpo \
  +algorithm.agent_version=copy \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=agents/agents.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=agent_hub \
  data.train_files="$TRAIN_FILES" data.val_files="$VAL_FILES" \
  data.train_batch_size=32 data.max_prompt_length=4096 data.max_response_length=2048 \
  actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
  actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.clip_ratio_low=0.20 actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  trainer.n_gpus_per_node="$N_GPUS" trainer.nnodes=1 \
  trainer.project_name=feynman26 trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUTPUT_DIR/$EXPERIMENT_NAME" \
  trainer.total_training_steps=200 trainer.total_epochs=10000 trainer.test_freq=5 trainer.save_freq=50
