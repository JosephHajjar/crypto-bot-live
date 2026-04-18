#!/bin/bash
# Auto-restarting wrapper for the 24/7 AI trade bot
# If the bot crashes, it waits 10 seconds and restarts automatically

restart_ensemble_bot() {
    while true; do
        echo "[BOT SUPERVISOR] Starting trade_live_alt_only.py..."
        python trade_live_alt_only.py
        EXIT_CODE=$?
        echo "[BOT SUPERVISOR] trade_live_alt_only.py exited with code $EXIT_CODE. Restarting in 10s..."
        sleep 10
    done
}

# Launch the auto-restarting ensemble bot in the background
restart_ensemble_bot &

# Start the dashboard/API web server as the main process
exec python -m gunicorn app:app -b 0.0.0.0:10000 --timeout 120 --workers 1 --threads 2
