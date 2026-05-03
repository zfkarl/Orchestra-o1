#!/usr/bin/env bash
# ========================================================================
# GRPO Training Orchestra-o1 Main Agent (Qwen3-8B) - Single node 8xH20 (96GB) full-parameter
# Pure training script: assumes expert_decisions.jsonl and train.parquet are pre-built.
#   For data preparation, run manually:
#     python3 build_expert_decisions.py --traj_dir ... --out ...
#     python3 prepare_data.py           --src ... --out_dir ... --expert_jsonl ... --require_expert
#
# VRAM budget (per GPU 96GB): see run_grpo_qwen3_8b.sh
# ========================================================================
set -euo pipefail

# ==================== Environment Configuration ====================
# Modify the following conda activation command for your environment
# source /path/to/your/miniconda3/bin/activate
# conda activate orchestra

# ---------- Path Configuration (modify for your environment) ----------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERL_DIR="${VERL_DIR:-/path/to/verl}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-8B}"

DATA_DIR="${DATA_DIR:-data/OmniGAIA/grpo_parquet}"
TRAIN_PARQUET="${DATA_DIR}/train.parquet"
EXPERT_JSONL="${DATA_DIR}/expert_decisions.jsonl"

GRPO_DIR="${PROJECT_DIR}/train_qwen3_8b/grpo"
REWARD_FN="${GRPO_DIR}/reward_fn.py"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="qwen3_8b_main_agent_grpo_${TIMESTAMP}"
PROJECT_NAME="grpo_orchestra_o1_qwen3_8b"

CKPT_DIR="${GRPO_DIR}/output/${EXP_NAME}"
LOG_DIR="${GRPO_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

export VERL_FILE_LOGGER_PATH="${CKPT_DIR}/metrics.jsonl"

# ==================== Wandb Configuration (optional) ====================
# export WANDB_API_KEY="your_wandb_api_key_here"
export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_NAME="${EXP_NAME}"

# ---------- Cluster Configuration ----------
NNODES=1
N_GPUS_PER_NODE=8

# ---------- Training Hyperparameters ----------
TRAIN_BSZ=${TRAIN_BSZ:-24}
MINI_BSZ=${MINI_BSZ:-12}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-24576}
MAX_RESP_LEN=${MAX_RESP_LEN:-4096}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-32768}
ROLLOUT_N=${ROLLOUT_N:-8}
LR=${LR:-5e-6}
EPOCHS=${EPOCHS:-5}
SAVE_FREQ=${SAVE_FREQ:-54}
TP_SIZE=${TP_SIZE:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.35}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}

# ---------- Pre-check: data must exist ----------
if [[ ! -f "${EXPERT_JSONL}" ]]; then
    echo "[train] ERROR: expert decisions not found: ${EXPERT_JSONL}"
    echo "[train]        please run build_expert_decisions.py first."
    exit 1
fi
if [[ ! -f "${TRAIN_PARQUET}" ]]; then
    echo "[train] ERROR: train parquet not found: ${TRAIN_PARQUET}"
    echo "[train]        please run prepare_data.py first."
    exit 1
fi
if ! head -n 1 "${EXPERT_JSONL}" | grep -q '"expert_steps"'; then
    echo "[train] ERROR: old expert_decisions.jsonl schema (no 'expert_steps' field)."
    echo "[train]        please rebuild via build_expert_decisions.py."
    exit 1
fi
echo "[train] expert: ${EXPERT_JSONL}"
echo "[train] parquet: ${TRAIN_PARQUET}"

# ---------- Enter verl directory and start training ----------
cd "${VERL_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${VERL_DIR}:${PYTHONPATH:-}"
export RAY_DEDUP_LOGS=0
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export TOKENIZERS_PARALLELISM=false

# ==================== Ray / NCCL Stability Configuration ====================
export RAY_worker_register_timeout_seconds=${RAY_worker_register_timeout_seconds:-600}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

TRAIN_LOG="${LOG_DIR}/train_${TIMESTAMP}.log"
echo "[train] log -> ${TRAIN_LOG}"
echo "[train] ckpt -> ${CKPT_DIR}"

set -x
set +e
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${TRAIN_PARQUET}" \
    data.train_batch_size=${TRAIN_BSZ} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESP_LEN} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="${REWARD_FN}" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb","file"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.default_local_dir="${CKPT_DIR}" \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","hf_model","optimizer","extra"]' \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=-1 \
    trainer.total_epochs=${EPOCHS} \
    ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS} \
    "$@" 2>&1 | tee "${TRAIN_LOG}"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set +x

echo ""
echo "=========================================="
if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
    echo "[train] ✅ Training completed successfully (exit code 0)"
else
    echo "[train] ⚠️  Training exited abnormally (exit code ${TRAIN_EXIT_CODE})"
fi
echo "[train] log: ${TRAIN_LOG}"
echo "=========================================="
echo "[train] All done."
