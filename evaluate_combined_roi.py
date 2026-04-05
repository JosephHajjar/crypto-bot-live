import json
import pandas as pd
import numpy as np
import datetime
import os
import sys

from dashboard import get_bot_signals, app, _load_bot_model

def evaluate_periods():
    print("Loading models and evaluating all history, please wait...")
    # Load all data using dashboard.py's internal function with a large limit
    with app.test_request_context('/api/bot_signals?limit=50000'):
        # This will use get_bot_signals which computes everything and returns signals
        res = get_bot_signals()
        data = res.json
        if "error" in data:
            print("API Error:", data["error"])
            return
            
        signals = data.get("signals", [])
        
    print(f"Total signals generated: {len(signals)}")
    buy_sigs = [s for s in signals if s.get("signal") == "BUY"]
    sell_sigs = [s for s in signals if s.get("signal") == "SELL"]
    close_sigs = [s for s in signals if s.get("signal") == "CLOSE"]
    print(f"Total BUYS: {len(buy_sigs)}, Total SELLS: {len(sell_sigs)}, Total CLOSES: {len(close_sigs)}")
    
    if len(close_sigs) == 0:
        print("No completed trades to evaluate.")
        return
        
    # Get the latest timestamp from signals to act as "now"
    latest_time = max([s['time'] for s in signals])
    now_dt = datetime.datetime.utcfromtimestamp(latest_time)
    print(f"Latest signal time: {now_dt} UTC")
    
    periods = [
        ("1 Day", 1),
        ("3 Days", 3),
        ("7 Days", 7),
        ("14 Days", 14),
        ("30 Days", 30),
        ("All Time", 9999)
    ]
    
    results = []
    
    for label, days in periods:
        cutoff_time = latest_time - (days * 24 * 3600) if days != 9999 else 0
        
        # We need to map closes back to their entries to know if they were Long or Short
        # But for simplicity, we can just look at PnL from the CLOSE signals that occurred within this period
        period_closes = [s for s in close_sigs if s['time'] >= cutoff_time]
        
        if len(period_closes) == 0:
            results.append({"Period": label, "Trades": 0, "Win Rate": "0%", "Total ROI": "0.00%", "Avg PnL": "0.00%"})
            continue
            
        wins = len([s for s in period_closes if s['pnl'] > 0])
        total_trades = len(period_closes)
        win_rate = (wins / total_trades) * 100
        total_pnl = sum([s['pnl'] for s in period_closes])
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        results.append({
            "Period": label,
            "Trades": total_trades,
            "Win Rate": f"{win_rate:.1f}%",
            "Total ROI": f"{total_pnl:.2f}%",
            "Avg Trade": f"{avg_pnl:.2f}%"
        })
        
    # Print formatted table
    df = pd.DataFrame(results)
    print("\n--- COMBINED DUAL-BOT PERFORMANCE ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    evaluate_periods()
