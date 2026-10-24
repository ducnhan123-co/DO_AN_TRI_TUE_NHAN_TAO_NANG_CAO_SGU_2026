#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PORT="${1:-8777}"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT đang bận. Dùng: ./run_ui.sh 8788"
  exit 1
fi

echo "Connect 4 RL Lab  →  http://127.0.0.1:$PORT/ui/"
python3 -m http.server "$PORT" --bind 127.0.0.1 &
PID=$!
trap 'kill "$PID" 2>/dev/null' EXIT

sleep 0.6
if command -v open >/dev/null 2>&1; then
  open "http://127.0.0.1:$PORT/ui/" 2>/dev/null || true
fi

wait "$PID"
