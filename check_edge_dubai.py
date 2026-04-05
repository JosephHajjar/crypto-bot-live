"""Check BTC AI Edge from 3:30 PM to 4:30 PM Dubai time."""
import pandas as pd
import numpy as np
import torch
import json
import requests
import pytz
from ml.model import AttentionLSTMModel
from data.feature_engineer_btc import compute_live_features, get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dubai = pytz.timezone("Asia/Dubai")

# Fetch raw data exactly like trade_live.py
print("Fetching live exchange data...")
url = "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=1000"
res = requests.get(url, timeout=15)
raw_candles = res.json()
candles = [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in raw_candles]
df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# Engineer features using LIVE path (no label truncation)
print("Engineering live features...")
live_df = compute_live_features(df, "data_storage/BTC_USDT_15m_scaler.json")

# Load models
with open("models/holy_grail_config.json", "r") as f: cfg_l = json.load(f)
m_long = AttentionLSTMModel(
    cfg_l['input_dim'], cfg_l['hidden_dim'], cfg_l['num_layers'], 2, cfg_l['dropout'], cfg_l['num_heads']
).to(device)
m_long.load_state_dict(torch.load("models/holy_grail.pth", map_location=device, weights_only=True))
m_long.eval()

with open("models_short/holy_grail_short_config.json", "r") as f: cfg_s = json.load(f)
m_short = AttentionLSTMModel(
    cfg_s['input_dim'], cfg_s['hidden_dim'], cfg_s['num_layers'], 2, cfg_s['dropout'], cfg_s['num_heads']
).to(device)
m_short.load_state_dict(torch.load("models_short/holy_grail_short.pth", map_location=device, weights_only=True))
m_short.eval()

seq_long = cfg_l.get('seq_len', 128)
seq_short = cfg_s.get('seq_len', 128)
cols = get_feature_cols()
feats = live_df[cols].values.astype(np.float32)

print("\n========= BTC AI EDGE: 3:30 PM - 4:30 PM DUBAI =========")
print(f"{'Time (Dubai)':>15}  |  {'LONG EDGE':>12}  |  {'SHORT EDGE':>12}  |  {'BTC Price':>10}")
print("-" * 65)

with torch.no_grad():
    for i in range(max(seq_long, seq_short), len(live_df)):
        dt_utc = live_df.index[i]
        dt_dubai = dt_utc.tz_localize('UTC').tz_convert(dubai)
        
        # Filter: only show 3:30 PM to 4:30 PM Dubai (= 11:30 to 12:30 UTC)
        h, m = dt_dubai.hour, dt_dubai.minute
        if not (h == 15 and m >= 30) and not (h == 16 and m <= 30):
            continue

        seq_l = feats[i - seq_long:i]
        seq_s = feats[i - seq_short:i]
        
        bull = torch.softmax(m_long(torch.tensor(np.array([seq_l])).to(device)), dim=1)[0][1].item() * 100
        bear = torch.softmax(m_short(torch.tensor(np.array([seq_s])).to(device)), dim=1)[0][1].item() * 100
        
        price = live_df.iloc[i]['close'] if 'close' in live_df.columns else 0
        
        time_str = dt_dubai.strftime("%I:%M %p")
        print(f"{time_str:>15}  |  {bull:>11.4f}%  |  {bear:>11.4f}%  |  ${price:>9.2f}")
