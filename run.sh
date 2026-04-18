#!/bin/bash
# Single-process architecture: bot runs as a thread inside gunicorn
# This halves RAM usage by sharing Python + PyTorch between dashboard and bot

# Set RENDER env var so app.py knows to auto-start the bot thread
export RENDER=true

# Start the dashboard (which auto-starts the bot thread internally)
exec python -m gunicorn app:app -b 0.0.0.0:10000 --timeout 120 --workers 1 --threads 4 --preload
