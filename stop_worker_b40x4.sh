#!/bin/bash
# Stop nnunet_worker_slurm running with .env.b40x4

ENV_FILE=".env.b40x4"

PIDS=()
for pid in $(pgrep -f "python main.py" 2>/dev/null); do
    if tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -qF "ENV_FILE=$ENV_FILE"; then
        PIDS+=("$pid")
    fi
done

if [ ${#PIDS[@]} -eq 0 ]; then
    echo "No worker found running with $ENV_FILE"
    exit 0
fi

for pid in "${PIDS[@]}"; do
    echo "Stopping worker PID $pid (ENV_FILE=$ENV_FILE)..."
    kill "$pid"
done

# Wait up to 10s for graceful shutdown
for i in $(seq 1 10); do
    sleep 1
    REMAINING=()
    for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && REMAINING+=("$pid")
    done
    PIDS=("${REMAINING[@]}")
    [ ${#PIDS[@]} -eq 0 ] && break
done

if [ ${#PIDS[@]} -gt 0 ]; then
    echo "Worker did not exit cleanly, sending SIGKILL..."
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null
    done
fi

echo "Done."
