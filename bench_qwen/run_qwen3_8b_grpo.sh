#!/bin/bash
# ============================================================
# One-click OmniGAIA Benchmark (Qwen3-8B GRPO as MainAgent)
#
# Function: auto-start vLLM -> wait ready -> run benchmark -> generate report
#
# Usage:
#   bash bench_qwen/run_qwen3_8b_grpo.sh
#
# Configuration can be overridden via environment variables:
#   CUDA_VISIBLE_DEVICES: Specify GPUs to use (e.g. "0", "0,1", "2,3")
#   VLLM_MODEL_PATH: Model path
#   VLLM_PORT: Service port (default 8803)
#   VLLM_TP_SIZE: tensor parallel size (default 1)
#   VLLM_GPU_MEMORY_UTILIZATION: GPU memory utilization (default 0.45)
#   VLLM_MAX_MODEL_LEN: Max model length (default 32768)
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 bash bench_qwen/run_qwen3_8b_grpo.sh
#   CUDA_VISIBLE_DEVICES=0,1 VLLM_TP_SIZE=2 bash bench_qwen/run_qwen3_8b_grpo.sh
# ============================================================

set -e

# # Disable Python output buffering (ensure real-time log writing with nohup)
export PYTHONUNBUFFERED=1

# # Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ==================== vLLM Configuration ====================
# Set VLLM_MODEL_PATH to your GRPO trained model path
MODEL_PATH="${VLLM_MODEL_PATH:-/path/to/your/grpo/checkpoint/huggingface}"
PORT="${VLLM_PORT:-8803}"
TP_SIZE="${VLLM_TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_LOG="${SCRIPT_DIR}/vllm_server_grpo.log"

# GPU configuration
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    GPU_INFO="GPU ${CUDA_VISIBLE_DEVICES}"
else
    GPU_INFO="All available GPUs"
fi

# # Cleanup function: auto-close vLLM server on script exit
cleanup() {
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Shutting down vLLM server (PID: ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null
        wait "${VLLM_PID}" 2>/dev/null || true
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] vLLM server stopped"
    fi
}
trap cleanup EXIT

echo "============================================================"
echo "  OmniGAIA Benchmark - Qwen3-8B GRPO MainAgent (One-click)"
echo "============================================================"
echo "  Project root:       ${PROJECT_ROOT}"
echo "  Model path:         ${MODEL_PATH}"
echo "  Service port:         ${PORT}"
echo "  Tensor Parallel:  ${TP_SIZE}"
echo "  GPU memory utilization:   ${GPU_MEMORY_UTILIZATION}"
echo "  Max sequence length:     ${MAX_MODEL_LEN}"
echo "  Using GPU:         ${GPU_INFO}"
echo "  vLLM log:         ${VLLM_LOG}"
echo "============================================================"

# Model path
if [ ! -d "${MODEL_PATH}" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: Model path does not exist: ${MODEL_PATH}"
    echo "  Please complete GRPO training first, or specify model path via VLLM_MODEL_PATH"
    exit 1
fi

# ==================== 1: Start vLLM server ====================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting vLLM server in background..."

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "qwen3-8b" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --dtype auto \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] vLLM server started (PID: ${VLLM_PID})"

# ==================== 2: Waiting for vLLM server to be ready ====================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for vLLM server to be ready..."
MAX_WAIT=2000
WAIT_INTERVAL=5
ELAPSED=0

while [ ${ELAPSED} -lt ${MAX_WAIT} ]; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: vLLM server exited abnormally! Check log: ${VLLM_LOG}"
        exit 1
    fi

    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] vLLM server is ready! (waited ${ELAPSED}s)"
        break
    fi

    sleep ${WAIT_INTERVAL}
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Still waiting for vLLM server... (${ELAPSED}/${MAX_WAIT}s)"
done

if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: vLLM server startup timed out (${MAX_WAIT}s)! Check log: ${VLLM_LOG}"
    exit 1
fi

# ==================== 3: Run Benchmark ====================
cd "${PROJECT_ROOT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting OmniGAIA Benchmark (Qwen3-8B GRPO)..."
python bench_qwen/bench_qwen_omnigaia.py \
    --config bench_qwen/orchestra_o1_omnigaia_qwen_grpo.yaml \
    --model_config bench_qwen/model_config_qwen.yaml \
    --vllm_url "http://localhost:${PORT}/v1/"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation completed!"

# ==================== 4: Generating evaluation report ====================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generating evaluation report..."
GRPO_OUTPUT_DIR="${PROJECT_ROOT}/logs/omnigaia_qwen_8b_grpo"
LATEST_CSV=$(ls -t "${GRPO_OUTPUT_DIR}"/omnigaia_qwen_*.csv 2>/dev/null | head -1)
if [ -n "${LATEST_CSV}" ]; then
    python bench_qwen/eval_qwen.py --csv_path "${LATEST_CSV}" --main_agent qwen3-8b-grpo
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: evaluation result CSV file not found (${GRPO_OUTPUT_DIR})"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done!"
