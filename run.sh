#!/bin/bash
# Start the live trading bot in the background (don't crash if it fails)
python -u trade_live.py &> /dev/null &

# Start the dashboard/API web server in the foreground
python -m gunicorn dashboard:app -b 0.0.0.0:10000 --timeout 120 --workers 1 --threads 2
