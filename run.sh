#!/bin/bash
# Start the dashboard/API web server (trade_live.py runs locally, not on Render)
exec python -m gunicorn app:app -b 0.0.0.0:10000 --timeout 120 --workers 1 --threads 2
