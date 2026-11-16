#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/Feynman26
export PATH="/root/miniconda3/bin:$PATH"

if [[ "${1:-}" != "--worker" ]]; then
  set -a
  source /root/autodl-tmp/.secrets/feynman26.env
  set +a
  python3 scripts/probe_judge_balance.py
  if tmux has-session -t feynman26 2>/dev/null; then
    echo "Training session feynman26 already exists." >&2
    exit 1
  fi
  tmux kill-session -t memmon 2>/dev/null || true
  tmux new-session -d -s memmon \
    "while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits >> outputs/gpu-memory.csv; sleep 1; done"
  tmux new-session -d -s feynman26 \
    "cd /root/autodl-tmp/Feynman26 && bash scripts/resume_lfm3_80g.sh --worker"
  tmux kill-session -t watchdog 2>/dev/null || true
  tmux new-session -d -s watchdog \
    "cd /root/autodl-tmp/Feynman26 && bash scripts/watch_feynman_run.sh"
  echo "Resumed in tmux session feynman26."
  exit 0
fi

export ACTOR_MODEL_PATH=/root/autodl-tmp/models/LFM3-1.7B-Uncensored
export TRAIN_FILES=data/feynman/feynman_grpo_v2_1500_train.parquet
export VAL_FILES=data/feynman/feynman_grpo_v2_1500_development.parquet
export OUTPUT_DIR=outputs
export EXPERIMENT_NAME=lfm3-1.7b-grpo-b64-80g
export N_GPUS=1
export TRAIN_BATCH_SIZE=64
export PPO_MINI_BATCH_SIZE=64
export ROLLOUT_N=8
export MAX_RESPONSE_LENGTH=1024
export TOTAL_TRAINING_STEPS=50
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export SAVE_FREQ=5
export TEST_FREQ=5
export ROLLOUT_NAME=hf
export ROLLOUT_MODE=sync
export PARAM_OFFLOAD=False
export OPTIMIZER_OFFLOAD=False
export MAX_ACTOR_CKPT_TO_KEEP=2
export FEYNMAN_REWARD_WORKERS=32
export ROLLOUT_DATA_DIR=outputs/evaluations/train
export VALIDATION_DATA_DIR=outputs/evaluations/validation
export LOG_VAL_GENERATIONS=10

exec bash run_feynman_grpo.sh >> outputs/feynman26.log 2>&1
