#!/bin/bash
# Feynman26 GRPO pilot. This script never downloads a model or dataset.
# Run it only after connecting to the training host and placing an SFT/midtrained
# LiquidAI/LFM3-1.7B checkpoint at ACTOR_MODEL_PATH.

set -euo pipefail

# Optional persistent secrets file. Keep it outside the repository and mode 600.
secrets_file="${FEYNMAN_SECRETS_FILE:-/root/autodl-tmp/.secrets/feynman26.env}"
if [[ -f "$secrets_file" ]]; then
  permissions="$(stat -c '%a' "$secrets_file")"
  if [[ "$permissions" != "600" ]]; then
    echo "Refusing to load $secrets_file: expected permissions 600, found $permissions." >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "$secrets_file"
  set +a
fi

MODEL_PLACEHOLDER="/path/to/feynman-sft-based-on-LiquidAI-LFM3-1.7B"
actor_model_path="${ACTOR_MODEL_PATH:-$MODEL_PLACEHOLDER}"
if [[ "$actor_model_path" == "$MODEL_PLACEHOLDER" ]] || [[ ! -e "$actor_model_path" ]]; then
  echo "Set ACTOR_MODEL_PATH to an existing Feynman SFT/midtrained LiquidAI/LFM3-1.7B checkpoint." >&2
  echo "No model will be downloaded. Intended base: https://huggingface.co/LiquidAI/LFM3-1.7B" >&2
  exit 1
fi

train_files="${TRAIN_FILES:-data/feynman/feynman_real100.parquet}"
val_files="${VAL_FILES:-}"
if [[ ! -f "$train_files" ]]; then
  echo "Training parquet not found: $train_files" >&2
  exit 1
fi
if [[ -z "$val_files" ]] || [[ ! -f "$val_files" ]]; then
  echo "Set VAL_FILES to a held-out parquet. The 100 verified pilot rows are all training rows and must not be reused as evaluation." >&2
  exit 1
fi

output_dir="${OUTPUT_DIR:-outputs}"
project_name="${FEYNMAN_SWANLAB_PROJECT:-${SWANLAB_PROJECT:-feynman}}"
# New SwanLab versions reserve SWANLAB_PROJECT for a structured settings value.
# Pass the project through VERL's trainer config instead of its environment parser.
unset SWANLAB_PROJECT
experiment_name="${EXPERIMENT_NAME:-lfm3-1.7b-grpo-pilot}"
n_gpus="${N_GPUS:-8}"
judge_model="${FEYNMAN_JUDGE_MODEL:-}"
if [[ -z "$judge_model" ]]; then
  echo "Set FEYNMAN_JUDGE_MODEL to the fixed, independent judge model for this run." >&2
  exit 1
fi

export FEYNMAN_JUDGE_MODEL="$judge_model"
export FEYNMAN_JUDGE_REASONING="${FEYNMAN_JUDGE_REASONING:-low}"
export FEYNMAN_JUDGE_CACHE="${FEYNMAN_JUDGE_CACHE:-$output_dir/judge_cache}"
export HF_HOME="$output_dir/hf_cache"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-$output_dir/swanlab}"
mkdir -p "$HF_HOME" "$SWANLAB_LOG_DIR" "$output_dir/$experiment_name"

NCCL_DEBUG=WARN python3 train_ppo.py \
  hydra.run.dir="$output_dir/hydra" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  +algorithm.agent_version=default \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=agents/agents.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=agent_hub \
  actor_rollout_ref.rollout.agent.num_workers=64 \
  data.train_files="$train_files" \
  data.val_files="$val_files" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-8}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-4096}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-1024}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  custom_reward_function.path=agents/feynman/verl_reward.py \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="$actor_model_path" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank="${LORA_RANK:-32}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA:-64}" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-8}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD:-True}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD:-True}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name="${ROLLOUT_NAME:-vllm}" \
  actor_rollout_ref.rollout.mode="${ROLLOUT_MODE:-async}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${INFER_TP:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.7}" \
  actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER:-True}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
  actor_rollout_ref.rollout.temperature="${POLICY_TEMPERATURE:-0.8}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  trainer.n_gpus_per_node="$n_gpus" \
  trainer.nnodes=1 \
  trainer.logger='["console","swanlab"]' \
  trainer.project_name="$project_name" \
  trainer.experiment_name="$experiment_name" \
  trainer.default_local_dir="$output_dir/$experiment_name" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR:-null}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR:-null}" \
  trainer.log_val_generations="${LOG_VAL_GENERATIONS:-0}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-2}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.test_freq="${TEST_FREQ:-5}" \
  trainer.save_freq="${SAVE_FREQ:-5}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-50}" \
  trainer.total_epochs=10000
