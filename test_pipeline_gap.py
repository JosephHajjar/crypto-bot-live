"""Test: Does compute_live_features drop the most recent candles?"""
import pandas as pd
import numpy as np
import requests

# Replicate exactly what trade_live.py does
url = "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=1000"
res = requests.get(url, timeout=15)
raw_candles = res.json()
candles = [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in raw_candles]
df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

print(f"Raw candle count: {len(df)}")
print(f"Latest raw candle: {df['timestamp'].iloc[-1]}")
print(f"Oldest raw candle: {df['timestamp'].iloc[0]}")

from data.feature_engineer_btc import compute_live_features
live_df = compute_live_features(df)
print(f"\nAfter compute_live_features: {len(live_df)} rows")
print(f"Latest FEATURE row : {live_df.index[-1]}")
print(f"Latest RAW candle  : {df['timestamp'].iloc[-1]}")

gap = df['timestamp'].iloc[-1] - live_df.index[-1]
print(f"\n>>> TIME GAP between raw data and features: {gap}")
if gap > pd.Timedelta(minutes=16):
    print("!!! CRITICAL: The live bot is evaluating STALE data !!!")
else:
    print("OK: Live bot is evaluating the most recent candle.")
