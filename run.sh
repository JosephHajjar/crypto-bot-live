#!/bin/bash
# Start the live trading bot in the background
python -u trade_live.py &

# Start the dashboard/API web server in the foreground
python -m gunicorn dashboard:app -b 0.0.0.0:10000 --timeout 120
