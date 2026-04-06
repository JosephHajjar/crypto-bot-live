#!/bin/bash
# Auto-restarting wrapper for the 24/7 AI trade bot
# If the bot crashes, it waits 10 seconds and restarts automatically

restart_bot() {
    while true; do
        echo "[BOT SUPERVISOR] Starting trade_live.py..."
        python trade_live.py
        EXIT_CODE=$?
        echo "[BOT SUPERVISOR] trade_live.py exited with code $EXIT_CODE. Restarting in 10s..."
        sleep 10
    done
}

restart_proportional_bot() {
    while true; do
        echo "[BOT SUPERVISOR] Starting trade_live_proportional.py..."
        python trade_live_proportional.py
        EXIT_CODE=$?
        echo "[BOT SUPERVISOR] trade_live_proportional.py exited with code $EXIT_CODE. Restarting in 10s..."
        sleep 10
    done
}

# Launch the auto-restarting bots in the background
restart_bot &
restart_proportional_bot &

# Start the dashboard/API web server as the main process
exec python -m gunicorn app:app -b 0.0.0.0:10000 --timeout 120 --workers 1 --threads 2
