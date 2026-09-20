#!/usr/bin/env bash
set -uo pipefail

MODEL=${MODEL:?set MODEL to the .gguf inside the container}
SERVED_NAME=${SERVED_NAME:-bonsai}
PORT=${PORT:-8010}
DECISION_PORT=${DECISION_PORT:-8011}
CTX=${CTX:-8192}
NGL=${NGL:-99}
PARALLEL=${PARALLEL:-1}
HEADROOM_GB=${HEADROOM_GB:-8}
DRAFT=${DRAFT:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [ ! -f "$MODEL" ]; then
  echo "no model at $MODEL" >&2
  exit 1
fi

# Unified memory: an overshoot takes the host down rather than the request.
avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
need=$(( $(stat -c %s "$MODEL") / 1073741824 + HEADROOM_GB ))
if [ "$avail" -lt "$need" ]; then
  echo "refusing to start: ${avail} GB available, need ~${need} GB" >&2
  exit 1
fi
echo "starting: ${avail} GB available, model $(basename "$MODEL")" >&2

draft_args=()
[ -n "$DRAFT" ] && [ -f "$DRAFT" ] && draft_args=(--model-draft "$DRAFT")

# shellcheck disable=SC2086
llama-server \
  --model "$MODEL" \
  --alias "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --ctx-size "$CTX" \
  --n-gpu-layers "$NGL" \
  --parallel "$PARALLEL" \
  --cache-reuse 256 \
  "${draft_args[@]}" \
  $EXTRA_ARGS &
LLAMA_PID=$!

for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 "$LLAMA_PID" 2>/dev/null || { echo "llama-server exited during start" >&2; exit 1; }
  sleep 2
done

# The decision server restarts on its own so its code can be swapped without
# reloading the weights: docker cp the file in, then pkill -f bjev_server.py
(
  while kill -0 "$LLAMA_PID" 2>/dev/null; do
    python3 /opt/bjev/bjev_server.py \
      --upstream "http://127.0.0.1:${PORT}" \
      --model "$SERVED_NAME" \
      --port "$DECISION_PORT" || true
    echo "decision server exited; restarting" >&2
    sleep 1
  done
) &
SERVER_LOOP=$!

trap 'kill "$LLAMA_PID" "$SERVER_LOOP" 2>/dev/null; pkill -f bjev_server.py 2>/dev/null; wait' TERM INT
wait "$LLAMA_PID"
kill "$SERVER_LOOP" 2>/dev/null
pkill -f bjev_server.py 2>/dev/null
wait
