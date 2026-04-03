#!/bin/bash
# Run the 24/7 AI trade bot continuously in the background
python trade_live.py &

# Start the dashboard/API web server concurrently
exec python -m gunicorn app:app -b 0.0.0.0:10000 --timeout 120 --workers 1 --threads 2
