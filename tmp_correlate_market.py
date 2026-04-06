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
    with open('c:/Users/asdf/.gemini/antigravity/scratch/ml_trading_bot/models/holy_grail_config.json', 'r') as f: cfg_l = json.load(f)
    ml = AttentionLSTMModel(
        input_dim=cfg_l['input_dim'], hidden_dim=cfg_l['hidden_dim'],
        num_layers=cfg_l['num_layers'], output_dim=2, dropout=cfg_l['dropout'], num_heads=cfg_l['num_heads']
    ).to(device)
    ml.load_state_dict(torch.load('c:/Users/asdf/.gemini/antigravity/scratch/ml_trading_bot/models/holy_grail.pth', map_location=device, weights_only=True))
    ml.eval()

    with open('c:/Users/asdf/.gemini/antigravity/scratch/ml_trading_bot/models_short/holy_grail_short_config.json', 'r') as f: cfg_s = json.load(f)
    ms = AttentionLSTMModel(
        input_dim=cfg_s['input_dim'], hidden_dim=cfg_s['hidden_dim'],
        num_layers=cfg_s['num_layers'], output_dim=2, dropout=cfg_s['dropout'], num_heads=cfg_s['num_heads']
    ).to(device)
    ms.load_state_dict(torch.load('c:/Users/asdf/.gemini/antigravity/scratch/ml_trading_bot/models_short/holy_grail_short.pth', map_location=device, weights_only=True))
    ms.eval()
    
    return ml, ms, cfg_l, cfg_s

ml, ms, cfg_l, cfg_s = load_models()
max_seq = max(cfg_l['seq_len'], cfg_s['seq_len'])

df = pd.read_csv('c:/Users/asdf/.gemini/antigravity/scratch/ml_trading_bot/data_storage/BTC_USDT_15m_processed.csv')

features = df[get_feature_cols()].values.astype(np.float32)
closes = df['close'].values
highs = df['high'].values
lows = df['low'].values
volumes = df['volume'].values

all_bull, all_bear = [], []
batch_size = 512
with torch.no_grad():
    for start in range(0, len(features) - max_seq + 1, batch_size):
        end = min(start + batch_size, len(features) - max_seq + 1)
        bl = [features[i + max_seq - cfg_l['seq_len'] : i + max_seq] for i in range(start, end)]
        bs = [features[i + max_seq - cfg_s['seq_len'] : i + max_seq] for i in range(start, end)]
        all_bull.extend(torch.softmax(ml(torch.tensor(np.array(bl)).to(device)), dim=1)[:, 1].cpu().tolist())
        all_bear.extend(torch.softmax(ms(torch.tensor(np.array(bs)).to(device)), dim=1)[:, 1].cpu().tolist())

closes = closes[max_seq - 1:]
highs = highs[max_seq - 1:]
lows = lows[max_seq - 1:]
volumes = volumes[max_seq - 1:]
all_bull = np.array(all_bull)
all_bear = np.array(all_bear)

chunk_size = 2880 # Approx 30 days
total_chunks = len(closes) // chunk_size

total_pnl_prop, total_trades_prop = 0.0, 0
total_pnl_alts, total_trades_alts = 0.0, 0

print(f"{'MO.':<4} | {'VOLATILITY':<10} | {'TREND %':<8} | {'AVG VOL':<8} | {'ALTs ROI':<9} | {'PROP ROI':<9} | {'PROP EDGE':<10}")
print("-" * 75)

for chunk_idx in range(total_chunks):
    start_i = chunk_idx * chunk_size
    end_i = start_i + chunk_size
    
    c_closes = closes[start_i:end_i]
    c_highs = highs[start_i:end_i]
    c_lows = lows[start_i:end_i]
    c_bull = all_bull[start_i:end_i]
    c_bear = all_bear[start_i:end_i]
    c_volumes = volumes[start_i:end_i]
    
    # Calculate Market Conditions
    returns = (c_closes[1:] - c_closes[:-1]) / c_closes[:-1]
    volatility = np.std(returns) * np.sqrt(2880) * 100 # Approx monthly volatility %
    trend_pct = (c_closes[-1] - c_closes[0]) / c_closes[0] * 100
    avg_vol = np.mean(c_volumes)
    
    # 1. Evaluate ALTs (Normal Model)
    pnl_alts = 0.0
    trades_alts = 0
    pos_alts = None
    ep_alts = 0.0
    tp_alts = 0.0
    sl_alts = 0.0
    bh_alts = 0
    
    for i in range(len(c_closes)):
        c, h, l, bull, bear = c_closes[i], c_highs[i], c_lows[i], c_bull[i], c_bear[i]
        
        if pos_alts is not None:
            bh_alts += 1
            hit_exit = False
            exit_price = c
            
            if pos_alts == 'long':
                if h >= tp_alts: hit_exit = True; exit_price = tp_alts
                elif l <= sl_alts: hit_exit = True; exit_price = sl_alts
                elif bh_alts >= cfg_l['max_hold_bars']: hit_exit = True; exit_price = c
                elif bear >= 0.50: hit_exit = True; exit_price = c 
            else:
                if l <= tp_alts: hit_exit = True; exit_price = tp_alts
                elif h >= sl_alts: hit_exit = True; exit_price = sl_alts
                elif bh_alts >= cfg_s['max_hold_bars']: hit_exit = True; exit_price = c
                elif bull >= 0.60: hit_exit = True; exit_price = c
                
            if hit_exit:
                ret = (exit_price - ep_alts) / ep_alts if pos_alts == 'long' else (ep_alts - exit_price) / ep_alts
                pnl_alts += (ret * 100) - 0.07 
                pos_alts = None
                bh_alts = 0
                ep_alts = 0.0
                
        if pos_alts is None:
            if bull >= 0.60:
                pos_alts = 'long'; ep_alts = c
                tp_alts = ep_alts * (1.0 + cfg_l['take_profit'])
                sl_alts = ep_alts * (1.0 - cfg_l['stop_loss'])
                trades_alts += 1
                bh_alts = 0
            elif bear >= 0.50:
                pos_alts = 'short'; ep_alts = c
                tp_alts = ep_alts * (1.0 - cfg_s['take_profit'])
                sl_alts = ep_alts * (1.0 + cfg_s['stop_loss'])
                trades_alts += 1
                bh_alts = 0

    # 2. Evaluate PROP (Optimized)
    pnl_prop = 0.0
    trades_prop = 0
    pos_prop = 'flat'
    ep_prop = 0.0

    for i in range(len(c_closes)):
        c, bull, bear = c_closes[i], c_bull[i], c_bear[i]
        d_bull = bull - bear
        d_bear = bear - bull
        
        if pos_prop == 'flat':
            if d_bull >= 0.2288:
                pos_prop = 'long'; ep_prop = c
            elif d_bear >= 0.2288:
                pos_prop = 'short'; ep_prop = c
        elif pos_prop == 'long':
            if d_bear >= 0.0008: 
                pnl_prop += ((c - ep_prop)/ep_prop * 100) - 0.07; trades_prop += 1
                pos_prop = 'short'; ep_prop = c
            elif d_bull < -0.0666: 
                pnl_prop += ((c - ep_prop)/ep_prop * 100) - 0.07; trades_prop += 1
                pos_prop = 'flat'; ep_prop = c
        elif pos_prop == 'short':
            if d_bull >= 0.0008: 
                pnl_prop += ((ep_prop - c)/ep_prop * 100) - 0.07; trades_prop += 1
                pos_prop = 'long'; ep_prop = c
            elif d_bear < -0.0666: 
                pnl_prop += ((ep_prop - c)/ep_prop * 100) - 0.07; trades_prop += 1
                pos_prop = 'flat'; ep_prop = c
                
    month_id = total_chunks - chunk_idx
    edge = pnl_prop - pnl_alts
    print(f"-{month_id:<3} | {volatility:>8.2f}% | {trend_pct:>+7.2f}% | {avg_vol:>7.0f} | {pnl_alts:>+8.2f}% | {pnl_prop:>+8.2f}% | {edge:>+9.2f}%")

print("-" * 75)
