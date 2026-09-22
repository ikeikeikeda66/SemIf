#!/bin/bash
# SemIf resident server launcher with volume wait support

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Wait up to 30 seconds for external drive/environment if needed
MAX_WAIT=30
WAITED=0
while [ ! -f "$DIR/.venv/bin/python" ] && [ $WAITED -lt $MAX_WAIT ]; do
    sleep 1
    WAITED=$((WAITED + 1))
done

if [ ! -f "$DIR/.venv/bin/python" ]; then
    echo "[$(date)] Error: Virtualenv python not found at $DIR/.venv/bin/python" >&2
    exit 1
fi

cd "$DIR" || exit 1
exec "$DIR/.venv/bin/python" "$DIR/scripts/semif_server.py" \
    --host "${SEMIF_HOST:-127.0.0.1}" \
    --port "${SEMIF_PORT:-8765}" \
    --device "${SEMIF_DEVICE:-mps}" \
    --dtype "${SEMIF_DTYPE:-float16}"
