#!/usr/bin/env bash
# * USAGE: ./run_vllm.sh [model] <port>

set -euo pipefail

MODEL="${1:?MODEL required (e.g. Qwen/Qwen3-32B)}"
PORT="${2:-11800}"

# ? API KEY
if [[ -f api_key ]]; then
    API_KEY=$(<api_key)
    API_KEY="${API_KEY//$'\n'/}"
else
    echo "[run_vllm] WARNING: api_key file not found"
    API_KEY=""
fi

# ? GPU / TP SIZE
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    TP_SIZE=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
    TP_SIZE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
fi

TP_SIZE=${TP_SIZE:-1}

# ^ PORT CHECK
if ss -ltn | grep -q ":$PORT "; then
    echo "[run_vllm] ERROR: port $PORT already in use"
    exit 1
fi

# ^ INFO
echo "[run_vllm] model=$MODEL"
echo "[run_vllm] port=$PORT"
echo "[run_vllm] tp_size=$TP_SIZE"
echo "[run_vllm] gpus=${CUDA_VISIBLE_DEVICES:-ALL}"

# ^ RUN
exec vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    ${API_KEY:+--api-key "$API_KEY"} \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
