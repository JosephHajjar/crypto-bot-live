import pandas as pd
import numpy as np
import torch
import json
import sys
import os

sys.path.insert(0, 'c:/Users/asdf/.gemini/antigravity/scratch/ml_trading_bot')
from ml.model import AttentionLSTMModel
from data.feature_engineer_btc import get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    # Long
    with open('c:/Users/asdf/.gemini/antigravity/scratch/ml_trading_bot/models/holy_grail_config.json', 'r') as f: cfg_l = json.load(f)
    ml = AttentionLSTMModel(
        input_dim=cfg_l['input_dim'], hidden_dim=cfg_l['hidden_dim'],
        num_layers=cfg_l['num_layers'], output_dim=2, dropout=cfg_l['dropout'], num_heads=cfg_l['num_heads']
    ).to(device)
    ml.load_state_dict(torch.load('c:/Users/asdf/.gemini/antigravity/scratch/ml_trading_bot/models/holy_grail.pth', map_location=device, weights_only=True))
    ml.eval()

    # Short
    with open('c:/Users/asdf/.gemini/antigravity/scratch/ml_trading_bot/models_short/holy_grail_short_config.json', 'r') as f: cfg_s = json.load(f)
    ms = AttentionLSTMModel(
        input_dim=cfg_s['input_dim'], hidden_dim=cfg_s['hidden_dim'],
        num_layers=cfg_s['num_layers'], output_dim=2, dropout=cfg_s['dropout'], num_heads=cfg_s['num_heads']
    ).to(device)
    ms.load_state_dict(torch.load('c:/Users/asdf/.gemini/antigravity/scratch/ml_trading_bot/models_short/holy_grail_short.pth', map_location=device, weights_only=True))
    ms.eval()
    
    return ml, ms, cfg_l, cfg_s

print("Loading models and 30 day data...")
ml, ms, cfg_l, cfg_s = load_models()
s_long = cfg_l['seq_len']
s_short = cfg_s['seq_len']
max_seq = max(s_long, s_short)

df = pd.read_csv('c:/Users/asdf/.gemini/antigravity/scratch/ml_trading_bot/data_storage/BTC_USDT_15m_processed.csv')
df = df.iloc[-2880:] # ~30 days

features = df[get_feature_cols()].values.astype(np.float32)
closes = df['close'].values
highs = df['high'].values
lows = df['low'].values

all_bull, all_bear = [], []
batch_size = 512
with torch.no_grad():
    for start in range(0, len(features) - max_seq + 1, batch_size):
        end = min(start + batch_size, len(features) - max_seq + 1)
        bl = [features[i + max_seq - s_long : i + max_seq] for i in range(start, end)]
        bs = [features[i + max_seq - s_short : i + max_seq] for i in range(start, end)]
        all_bull.extend(torch.softmax(ml(torch.tensor(np.array(bl)).to(device)), dim=1)[:, 1].cpu().tolist())
        all_bear.extend(torch.softmax(ms(torch.tensor(np.array(bs)).to(device)), dim=1)[:, 1].cpu().tolist())

# Adjust arrays
closes = closes[max_seq - 1:]
highs = highs[max_seq - 1:]
lows = lows[max_seq - 1:]
all_bull = np.array(all_bull)
all_bear = np.array(all_bear)

# SIMULATOR 1: ALTs
print("\n--- SIMULATING NORMAL AI (ALTs) ---")
pnl_alts = 0.0
trades_alts = 0
pos = None
ep = 0.0
tp = 0.0
sl = 0.0
bars_held = 0

for i in range(len(closes)):
    c = closes[i]
    h = highs[i]
    l = lows[i]
    
    bull_prob = all_bull[i]
    bear_prob = all_bear[i]
    
    if pos is not None:
        bars_held += 1
        
        hit_exit = False
        exit_price = c
        
        if pos == 'long':
            if h >= tp: hit_exit = True; exit_price = tp
            elif l <= sl: hit_exit = True; exit_price = sl
            elif bars_held >= cfg_l['max_hold_bars']: hit_exit = True; exit_price = c
            elif bear_prob >= 0.50: hit_exit = True; exit_price = c 
        else:
            if l <= tp: hit_exit = True; exit_price = tp
            elif h >= sl: hit_exit = True; exit_price = sl
            elif bars_held >= cfg_s['max_hold_bars']: hit_exit = True; exit_price = c
            elif bull_prob >= 0.60: hit_exit = True; exit_price = c
            
        if hit_exit:
            ret = (exit_price - ep) / ep if pos == 'long' else (ep - exit_price) / ep
            pnl_alts += (ret * 100) - 0.07 # -0.07% fees
            pos = None
            bars_held = 0
            ep = 0.0
            
    # Entry
    if pos is None:
        if bull_prob >= 0.60:
            pos = 'long'
            ep = c
            tp = ep * (1.0 + cfg_l['take_profit'])
            sl = ep * (1.0 - cfg_l['stop_loss'])
            trades_alts += 1
            bars_held = 0
        elif bear_prob >= 0.50:
            pos = 'short'
            ep = c
            tp = ep * (1.0 - cfg_s['take_profit'])
            sl = ep * (1.0 + cfg_s['stop_loss'])
            trades_alts += 1
            bars_held = 0

print(f"ALTs Net ROI: {pnl_alts:.2f}%")
print(f"ALTs Trades: {trades_alts}")


# SIMULATOR 2: PROP 30-Day Optimized
print("\n--- SIMULATING PROP (New 30-Day Settings) ---")
pnl_prop = 0.0
trades_prop = 0
pos = 'flat'
ep = 0.0

for i in range(len(closes)):
    c = closes[i]
    bull = all_bull[i]
    bear = all_bear[i]
    d_bull = bull - bear
    d_bear = bear - bull
    
    if pos == 'flat':
        if d_bull >= 0.2288:
            pos = 'long'; ep = c
        elif d_bear >= 0.2288:
            pos = 'short'; ep = c
    elif pos == 'long':
        if d_bear >= 0.0008: 
            pnl_prop += ((c - ep)/ep * 100) - 0.07; trades_prop += 1
            pos = 'short'; ep = c
        elif d_bull < -0.0666: 
            pnl_prop += ((c - ep)/ep * 100) - 0.07; trades_prop += 1
            pos = 'flat'; ep = c
    elif pos == 'short':
        if d_bull >= 0.0008: 
            pnl_prop += ((ep - c)/ep * 100) - 0.07; trades_prop += 1
            pos = 'long'; ep = c
        elif d_bear < -0.0666: 
            pnl_prop += ((ep - c)/ep * 100) - 0.07; trades_prop += 1
            pos = 'flat'; ep = c

print(f"PROP Net ROI: {pnl_prop:.2f}%")
print(f"PROP Trades: {trades_prop}")
