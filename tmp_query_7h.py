import pandas as pd
import numpy as np
import requests
import torch
import json
import sys
import os

# Add root to pythonpath
sys.path.insert(0, r'c:\Users\asdf\.gemini\antigravity\scratch\ml_trading_bot')
from data.feature_engineer_btc import compute_live_features, get_feature_cols
from ml.model import AttentionLSTMModel

os.chdir(r'c:\Users\asdf\.gemini\antigravity\scratch\ml_trading_bot')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fetch_recent_data():
    url = "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=1000"
    res = requests.get(url, timeout=15)
    raw_candles = res.json()
    candles = [
        [int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
        for c in raw_candles
    ]
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

try:
    df = fetch_recent_data()
except Exception as e:
    print(f"Error fetching data: {e}")
    sys.exit(1)

live_df = compute_live_features(df, 'data_storage/BTC_USDT_15m_scaler.json')

with open('models/holy_grail_config.json', 'r') as f:
    cfg_long = json.load(f)
with open('models_short/holy_grail_short_config.json', 'r') as f:
    cfg_short = json.load(f)

seq_len_long = cfg_long.get('seq_len', 128)
seq_len_short = cfg_short.get('seq_len', 128)

torch.serialization.add_safe_globals([AttentionLSTMModel])
model_long = AttentionLSTMModel(
    input_dim=cfg_long['input_dim'], hidden_dim=cfg_long['hidden_dim'],
    num_layers=cfg_long['num_layers'], output_dim=2, dropout=cfg_long['dropout'], num_heads=cfg_long['num_heads']
).to(device)
model_long.load_state_dict(torch.load('models/holy_grail.pth', map_location=device, weights_only=True))
model_long.eval()

model_short = AttentionLSTMModel(
    input_dim=cfg_short['input_dim'], hidden_dim=cfg_short['hidden_dim'],
    num_layers=cfg_short['num_layers'], output_dim=2, dropout=cfg_short['dropout'], num_heads=cfg_short['num_heads']
).to(device)
model_short.load_state_dict(torch.load('models_short/holy_grail_short.pth', map_location=device, weights_only=True))
model_short.eval()

feature_cols = get_feature_cols()
feat_np = live_df[feature_cols].values.astype(np.float32)

start_idx = max(seq_len_long, seq_len_short)

if start_idx >= len(live_df):
    print("Not enough data fetched.")
    sys.exit(1)

num_trades = 0
in_trade = False
bars_held = 0
trade_type = None
entry_price = 0.0

long_tp = cfg_long.get('take_profit', 0.0125)
long_sl = cfg_long.get('stop_loss', 0.0250)
long_max_hold = cfg_long.get('max_hold_bars', 12)

short_tp = cfg_short.get('take_profit', 0.0150)
short_sl = cfg_short.get('stop_loss', 0.0080)
short_max_hold = cfg_short.get('max_hold_bars', 8)

max_bull_prob = 0.0
max_bear_prob = 0.0

for i in range(start_idx, len(live_df)):
    current_close = live_df['close'].iloc[i]
    current_high = live_df['high'].iloc[i]
    current_low = live_df['low'].iloc[i]
    
    if in_trade:
        bars_held += 1
        exit_price = None
        
        if trade_type == "LONG":
            tp_price = entry_price * (1 + long_tp)
            sl_price = entry_price * (1 - long_sl)
            max_bars = long_max_hold
            
            if current_low <= sl_price:
                exit_price = sl_price
            elif current_high >= tp_price:
                exit_price = tp_price
            elif bars_held >= max_bars:
                exit_price = current_close
        else: # SHORT
            tp_price = entry_price * (1 - short_tp)
            sl_price = entry_price * (1 + short_sl)
            max_bars = short_max_hold
            
            if current_high >= sl_price:
                exit_price = sl_price
            elif current_low <= tp_price:
                exit_price = tp_price
            elif bars_held >= max_bars:
                exit_price = current_close
                
        if exit_price is not None:
            ret = (exit_price - entry_price) / entry_price if trade_type == "LONG" else (entry_price - exit_price) / entry_price
            ret_pct = ret * 100
            real_timestamp_exit = df['timestamp'].iloc[-len(live_df) + i]
            print(f"[{real_timestamp_exit}] {trade_type} CLOSED @ {exit_price:.2f} | PNL: {ret_pct:+.2f}%")
            in_trade = False
            trade_type = None
            bars_held = 0
    
    feat_long = feat_np[i - seq_len_long + 1 : i + 1]
    feat_short = feat_np[i - seq_len_short + 1 : i + 1]
    
    tensor_long = torch.tensor(feat_long).unsqueeze(0).to(device)
    tensor_short = torch.tensor(feat_short).unsqueeze(0).to(device)
    
    with torch.no_grad():
        bull_prob = torch.softmax(model_long(tensor_long), dim=1)[0][1].item()
        bear_prob = torch.softmax(model_short(tensor_short), dim=1)[0][1].item()
        
    max_bull_prob = max(max_bull_prob, bull_prob)
    max_bear_prob = max(max_bear_prob, bear_prob)
        
    if not in_trade:
        if bull_prob >= 0.60 and bear_prob >= 0.50:
            pass
        elif bull_prob >= 0.60:
            in_trade = True
            trade_type = "LONG"
            entry_price = current_close
            bars_held = 0
            num_trades += 1
            real_timestamp = df['timestamp'].iloc[-len(live_df) + i]
            print(f"[{real_timestamp}] LONG entered")
        elif bear_prob >= 0.50:
            in_trade = True
            trade_type = "SHORT"
            entry_price = current_close
            bars_held = 0
            num_trades += 1
            real_timestamp = df['timestamp'].iloc[-len(live_df) + i]
            print(f"[{real_timestamp}] SHORT entered")

total_candles_checked = len(live_df) - start_idx
days_checked = total_candles_checked / 96.0

print(f"\n=======================")
print(f"Total trades opened in the past {days_checked:.2f} DAYS: {num_trades}")
print(f"Max Long probability: {max_bull_prob:.4f}")
print(f"Max Short probability: {max_bear_prob:.4f}")
print(f"That is an average of {num_trades / days_checked:.2f} trades per day.")
