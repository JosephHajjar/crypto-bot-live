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

print("Loading UNSEEN data (PAXG Gold 15m) for Bitcoin AI simulation...")
df = pd.read_csv('c:/Users/asdf/.gemini/antigravity/scratch/ml_trading_bot/data_storage/PAXG_USDT_15m_processed.csv')

features = df[get_feature_cols()].values.astype(np.float32)
closes = df['close'].values
highs = df['high'].values
lows = df['low'].values

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
all_bull = np.array(all_bull)
all_bear = np.array(all_bear)

total_pnl_prop, total_trades_prop = 0.0, 0
total_pnl_alts, total_trades_alts = 0.0, 0
total_pnl_dyn, total_trades_dyn = 0.0, 0

print(f"{'MONTH':<10} | {'NORMAL (ALTs)':<18} | {'GAP (PROP)':<18} | {'DYNAMIC (AUTO-SWITCH)':<18}")
print("-" * 75)

chunk_size = 2880 # Approx 30 days
total_chunks = len(closes) // chunk_size

for chunk_idx in range(total_chunks):
    start_i = chunk_idx * chunk_size
    end_i = start_i + chunk_size
    
    if start_i < 1000:
        start_i = 1000 # Need at least 1000 candles before this chunk for volatility calcs across chunk bounds
        if end_i <= start_i: continue
        
    c_closes = closes[start_i:end_i]
    c_highs = highs[start_i:end_i]
    c_lows = lows[start_i:end_i]
    c_bull = all_bull[start_i:end_i]
    c_bear = all_bear[start_i:end_i]
    
    # 1. Evaluate NORMAL (ALTs)
    pnl_alts, trades_alts = 0.0, 0
    pos_alts = None; ep_alts, tp_alts, sl_alts, bh_alts = 0.0, 0.0, 0.0, 0
    
    for i in range(len(c_closes)):
        c, h, l, bull, bear = c_closes[i], c_highs[i], c_lows[i], c_bull[i], c_bear[i]
        if pos_alts is not None:
            bh_alts += 1; hit_exit = False; exit_price = c
            if pos_alts == 'long':
                if h >= tp_alts: hit_exit = True; exit_price = tp_alts
                elif l <= sl_alts: hit_exit = True; exit_price = sl_alts
                elif bh_alts >= cfg_l['max_hold_bars']: hit_exit = True
                elif bear >= 0.50: hit_exit = True
            else:
                if l <= tp_alts: hit_exit = True; exit_price = tp_alts
                elif h >= sl_alts: hit_exit = True; exit_price = sl_alts
                elif bh_alts >= cfg_s['max_hold_bars']: hit_exit = True
                elif bull >= 0.60: hit_exit = True
            if hit_exit:
                pnl_alts += (((exit_price - ep_alts)/ep_alts if pos_alts == 'long' else (ep_alts - exit_price)/ep_alts) * 100) - 0.07 
                pos_alts = None; bh_alts = 0; ep_alts = 0.0
                
        if pos_alts is None:
            if bull >= 0.60:
                pos_alts = 'long'; ep_alts = c; tp_alts = c * (1.0 + cfg_l['take_profit']); sl_alts = c * (1.0 - cfg_l['stop_loss'])
                trades_alts += 1; bh_alts = 0
            elif bear >= 0.50:
                pos_alts = 'short'; ep_alts = c; tp_alts = c * (1.0 - cfg_s['take_profit']); sl_alts = c * (1.0 + cfg_s['stop_loss'])
                trades_alts += 1; bh_alts = 0

    # 2. Evaluate GAP (PROP)
    pnl_prop, trades_prop = 0.0, 0
    pos_prop, ep_prop = 'flat', 0.0
    for i in range(len(c_closes)):
        c, d_bull, d_bear = c_closes[i], c_bull[i]-c_bear[i], c_bear[i]-c_bull[i]
        if pos_prop == 'flat':
            if d_bull >= 0.2288: pos_prop = 'long'; ep_prop = c
            elif d_bear >= 0.2288: pos_prop = 'short'; ep_prop = c
        elif pos_prop == 'long':
            if d_bear >= 0.0008: pnl_prop += ((c-ep_prop)/ep_prop*100)-0.07; trades_prop += 1; pos_prop = 'short'; ep_prop = c
            elif d_bull < -0.0666: pnl_prop += ((c-ep_prop)/ep_prop*100)-0.07; trades_prop += 1; pos_prop = 'flat'; ep_prop = c
        elif pos_prop == 'short':
            if d_bull >= 0.0008: pnl_prop += ((ep_prop-c)/ep_prop*100)-0.07; trades_prop += 1; pos_prop = 'long'; ep_prop = c
            elif d_bear < -0.0666: pnl_prop += ((ep_prop-c)/ep_prop*100)-0.07; trades_prop += 1; pos_prop = 'flat'; ep_prop = c

    # 3. Evaluate DYNAMIC REGIME (Auto-Switch)
    pnl_dyn, trades_dyn = 0.0, 0
    master_control = 'ALT'
    pos_dyn = None; ep_dyn = 0.0; tp_dyn, sl_dyn = 0.0, 0.0; bh_dyn = 0
    
    for i in range(len(c_closes)):
        c, h, l, bull, bear = c_closes[i], c_highs[i], c_lows[i], c_bull[i], c_bear[i]
        d_bull, d_bear = bull - bear, bear - bull
        
        # Regime Update (only when flat)
        if pos_dyn is None or pos_dyn == 'flat':
            glob_i = start_i + i
            past_1000 = closes[glob_i - 1000 : glob_i]
            rets = (past_1000[1:] - past_1000[:-1]) / past_1000[:-1]
            vol_m = np.std(rets) * np.sqrt(2880) * 100
            master_control = 'PROP' if vol_m >= 10.50 else 'ALT'
            
        # Strategy Execution
        if master_control == 'ALT':
            if pos_dyn == 'flat': pos_dyn = None # Normalize
            if pos_dyn is not None:
                bh_dyn += 1; hit_exit = False; exit_price = c
                if pos_dyn == 'long':
                    if h >= tp_dyn: hit_exit = True; exit_price = tp_dyn
                    elif l <= sl_dyn: hit_exit = True; exit_price = sl_dyn
                    elif bh_dyn >= cfg_l['max_hold_bars']: hit_exit = True
                    elif bear >= 0.50: hit_exit = True
                else:
                    if l <= tp_dyn: hit_exit = True; exit_price = tp_dyn
                    elif h >= sl_dyn: hit_exit = True; exit_price = sl_dyn
                    elif bh_dyn >= cfg_s['max_hold_bars']: hit_exit = True
                    elif bull >= 0.60: hit_exit = True
                if hit_exit:
                    pnl_dyn += (((exit_price - ep_dyn)/ep_dyn if pos_dyn == 'long' else (ep_dyn - exit_price)/ep_dyn) * 100) - 0.07 
                    pos_dyn = None; bh_dyn = 0; ep_dyn = 0.0
            if pos_dyn is None:
                if bull >= 0.60:
                    pos_dyn = 'long'; ep_dyn = c; tp_dyn = c * (1.0 + cfg_l['take_profit']); sl_dyn = c * (1.0 - cfg_l['stop_loss'])
                    trades_dyn += 1; bh_dyn = 0
                elif bear >= 0.50:
                    pos_dyn = 'short'; ep_dyn = c; tp_dyn = c * (1.0 - cfg_s['take_profit']); sl_dyn = c * (1.0 + cfg_s['stop_loss'])
                    trades_dyn += 1; bh_dyn = 0
                    
        elif master_control == 'PROP':
            if pos_dyn is None: pos_dyn = 'flat' # Normalize
            if pos_dyn == 'flat':
                if d_bull >= 0.2288: pos_dyn = 'long'; ep_dyn = c
                elif d_bear >= 0.2288: pos_dyn = 'short'; ep_dyn = c
            elif pos_dyn == 'long':
                if d_bear >= 0.0008: pnl_dyn += ((c-ep_dyn)/ep_dyn*100)-0.07; trades_dyn += 1; pos_dyn = 'short'; ep_dyn = c
                elif d_bull < -0.0666: pnl_dyn += ((c-ep_dyn)/ep_dyn*100)-0.07; trades_dyn += 1; pos_dyn = 'flat'; ep_dyn = c
            elif pos_dyn == 'short':
                if d_bull >= 0.0008: pnl_dyn += ((ep_dyn-c)/ep_dyn*100)-0.07; trades_dyn += 1; pos_dyn = 'long'; ep_dyn = c
                elif d_bear < -0.0666: pnl_dyn += ((ep_dyn-c)/ep_dyn*100)-0.07; trades_dyn += 1; pos_dyn = 'flat'; ep_dyn = c

    month_id = total_chunks - chunk_idx
    print(f"Month {-month_id:<5} | {pnl_alts:>+7.2f}% ({trades_alts:>3}t) {'' if pnl_alts < max(pnl_prop,pnl_dyn,pnl_alts) else 'W '} | {pnl_prop:>+7.2f}% ({trades_prop:>3}t) {'' if pnl_prop < max(pnl_alts,pnl_prop,pnl_dyn) else 'W '} | {pnl_dyn:>+7.2f}% ({trades_dyn:>3}t) {'' if pnl_dyn < max(pnl_alts,pnl_prop,pnl_dyn) else 'W '}")
    
    total_pnl_alts += pnl_alts; total_trades_alts += trades_alts
    total_pnl_prop += pnl_prop; total_trades_prop += trades_prop
    total_pnl_dyn += pnl_dyn; total_trades_dyn += trades_dyn

print("-" * 75)
print(f"TOTAL      | {total_pnl_alts:>+7.2f}% ({total_trades_alts:>4}t) | {total_pnl_prop:>+7.2f}% ({total_trades_prop:>4}t) | {total_pnl_dyn:>+7.2f}% ({total_trades_dyn:>4}t)")
