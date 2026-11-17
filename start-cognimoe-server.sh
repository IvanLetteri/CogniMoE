#!/usr/bin/env bash
# start-cognimoe-server.sh — Wrench-35B-A3B Q4_K_M su CogniMoE (RTX 2070 Super 8 GB + PCIe Gen1)

set -u

COGNIMOE_DIR="$HOME/CogniMoE"
BIN="$COGNIMOE_DIR/build/bin/llama-server"
MODEL="$HOME/models/Wrench-35B-A3B-Q4_K_M.gguf"
WIKI_DIR="$COGNIMOE_DIR/cognimoe_wiki"

SKILL="${SKILL:-coding}"
CACHE_MIB="${CACHE_MIB:-3400}"
CPU_SHARE="${CPU_SHARE:-0.8}"
PENALTY="${PENALTY:-0.25}"
PORT="${PORT:-8080}"
MIN_FREE_MIB=5000

[[ -x "$BIN" ]]   || { echo "[ERR] Binario CogniMoE non trovato: $BIN"; exit 1; }
[[ -f "$MODEL" ]] || { echo "[ERR] Modello non trovato: $MODEL"; exit 1; }

if pgrep -f "llama-server" >/dev/null 2>&1; then
  echo "[INFO] Fermo il server llama-server precedente..."
  pkill -f "llama-server" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    pgrep -f "llama-server" >/dev/null 2>&1 || break
    sleep 1
  done
  pkill -9 -f "llama-server" 2>/dev/null || true
  sleep 2
fi

FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
if ! [[ "$FREE" =~ ^[0-9]+$ ]] || (( FREE < MIN_FREE_MIB )); then
  echo "[ERR] VRAM libera insufficiente (${FREE:-0} MiB)."
  exit 1
fi

PROFILE_FILE="$WIKI_DIR/profile_${SKILL}.json"
if [[ "$SKILL" != "none" && -f "$PROFILE_FILE" ]]; then
  export COGNIMOE_PREWARM_JSON="$PROFILE_FILE"
  export COGNIMOE_CACHE_MISS_PENALTY="$PENALTY"
  PREWARM_MSG="ATTIVO ($PROFILE_FILE | Penalty: $PENALTY)"
else
  unset COGNIMOE_PREWARM_JSON
  unset COGNIMOE_CACHE_MISS_PENALTY
  PREWARM_MSG="DISATTIVO (Cold Cache)"
fi

echo "=================================================================="
echo "[CogniMoE] Avvio Server Cognitivo per Hermes Agent"
echo "[INFO] VRAM libera: ${FREE} MiB | Cache VRAM: ${CACHE_MIB} MiB | CPU Share: ${CPU_SHARE}"
echo "[INFO] Pre-Warming & Soft-Masking: ${PREWARM_MSG} | Porta: ${PORT}"
echo "=================================================================="

ARGS=(
  -m "$MODEL"
  --alias Wrench-35B-A3B-Q4_K_M
  --jinja
  -ngl 99
  --cpu-moe
  --moe-cache-mib "$CACHE_MIB"
  --moe-cache-cpu-share "$CPU_SHARE"
  -c 65536
  -np 1
  -b 512 -ub 512
  --threads 6
  --threads-batch 6
  -fa on
  --cache-type-k q8_0
  --cache-type-v q8_0
  -lm none
  --host 0.0.0.0
  --port "$PORT"
)

exec "$BIN" "${ARGS[@]}"
