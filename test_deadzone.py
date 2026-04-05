import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ml.model import AttentionLSTMModel
from data.feature_engineer import get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    df = pd.read_csv('data_storage/BTC_USDT_15m_processed.csv')
    df = df.copy() 
    
    with open('models/holy_grail_config.json', 'r') as f: cfg_l = json.load(f)
    s_long = cfg_l.get('seq_len', 128)
    m_long = AttentionLSTMModel(
        input_dim=cfg_l['input_dim'], hidden_dim=cfg_l['hidden_dim'],
        num_layers=cfg_l['num_layers'], output_dim=2, dropout=cfg_l['dropout'], num_heads=cfg_l['num_heads']
    ).to(device)
    m_long.load_state_dict(torch.load('models/holy_grail.pth', map_location=device, weights_only=True))
    m_long.eval()

    with open('models_short/holy_grail_short_config.json', 'r') as f: cfg_s = json.load(f)
    s_short = cfg_s.get('seq_len', 128)
    m_short = AttentionLSTMModel(
        input_dim=cfg_s['input_dim'], hidden_dim=cfg_s['hidden_dim'],
        num_layers=cfg_s['num_layers'], output_dim=2, dropout=cfg_s['dropout'], num_heads=cfg_s['num_heads']
    ).to(device)
    m_short.load_state_dict(torch.load('models_short/holy_grail_short.pth', map_location=device, weights_only=True))
    m_short.eval()

    MAX_SEQ_LEN = max(s_long, s_short)
    feature_cols = get_feature_cols()
    unix_times = df['timestamp'].values
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    features_np = df[feature_cols].values.astype(np.float32)
    
    batch_size = 256
    all_bull, all_bear = [], []
    
    with torch.no_grad():
        for start in range(0, len(features_np) - MAX_SEQ_LEN + 1, batch_size):
            end = min(start + batch_size, len(features_np) - MAX_SEQ_LEN + 1)
            batch_l = [features_np[i + MAX_SEQ_LEN - s_long : i + MAX_SEQ_LEN] for i in range(start, end)]
            batch_s = [features_np[i + MAX_SEQ_LEN - s_short : i + MAX_SEQ_LEN] for i in range(start, end)]
            t_long = torch.tensor(np.array(batch_l)).to(device)
            t_short = torch.tensor(np.array(batch_s)).to(device)
            all_bull.extend(torch.softmax(m_long(t_long), dim=1)[:, 1].cpu().numpy().tolist())
            all_bear.extend(torch.softmax(m_short(t_short), dim=1)[:, 1].cpu().numpy().tolist())
            
    return unix_times, close_prices, high_prices, low_prices, all_bull, all_bear, MAX_SEQ_LEN

def evaluate_logic(use_deadzone, position_type, times, close_prices, high_prices, low_prices, all_bull, all_bear, max_seq):
    position = 'flat'
    entry_price = 0
    bars_held = 0
    trades = []
    
    for idx in range(len(all_bull)):
        bar_idx = max_seq - 1 + idx
        if bar_idx < len(times):
            time_val = times[bar_idx]
            dt = pd.to_datetime(time_val) if isinstance(time_val, str) else datetime.utcfromtimestamp(time_val)
            
            can_trade = True
            if use_deadzone:
                can_trade = not (13 <= dt.hour <= 16)
                
            c_p, h_p, l_p = close_prices[bar_idx], high_prices[bar_idx], low_prices[bar_idx]
            prob_bull, prob_bear = all_bull[idx], all_bear[idx]
            
            if position == 'flat':
                if prob_bull >= 0.60 and prob_bear >= 0.50 and can_trade: pass
                elif prob_bull >= 0.60 and can_trade and position_type == 'LONG':
                    position = 'long'; entry_price = c_p; bars_held = 0
                elif prob_bear >= 0.50 and can_trade and position_type == 'SHORT':
                    position = 'short'; entry_price = c_p; bars_held = 0
            else:
                bars_held += 1
                closed, pnl = False, 0
                if position == 'long':
                    # 1.25% TP, 2.5% SL, 12 bars
                    tp_p, sl_p = entry_price * 1.0125, entry_price * 0.975
                    if l_p <= sl_p: pnl = (sl_p - entry_price)/entry_price*100; closed=True
                    elif h_p >= tp_p: pnl = (tp_p - entry_price)/entry_price*100; closed=True
                    elif bars_held >= 12: pnl = (c_p - entry_price)/entry_price*100; closed=True
                elif position == 'short':
                    # 1.75% TP, 3.75% SL, 12 bars
                    tp_p, sl_p = entry_price * 0.9825, entry_price * 1.0375
                    if h_p >= sl_p: pnl = (entry_price - sl_p)/entry_price*100; closed=True
                    elif l_p <= tp_p: pnl = (entry_price - tp_p)/entry_price*100; closed=True
                    elif bars_held >= 12: pnl = (entry_price - c_p)/entry_price*100; closed=True
                        
                if closed:
                    trades.append({'pnl': pnl})
                    position = 'flat'
    return trades

def report(name, trades):
    if not trades: return
    df = pd.DataFrame(trades)
    win_rate = (df['pnl'] > 0).mean() * 100
    print(f"[{name:20}] -> | ROI: {df['pnl'].sum():7.2f}% | Win Rate: {win_rate:5.1f}% | Trades: {len(trades)}")

def main():
    u, c, h, l, bull, bear, m = load_data()
    print("\n--- LONG EVALUATION ---")
    report("LONG (Block 13h-16h)", evaluate_logic(True, 'LONG', u, c, h, l, bull, bear, m))
    report("LONG (Trade 24/7)", evaluate_logic(False, 'LONG', u, c, h, l, bull, bear, m))
    print("\n--- SHORT EVALUATION ---")
    report("SHORT (Block 13h-16h)", evaluate_logic(True, 'SHORT', u, c, h, l, bull, bear, m))
    report("SHORT (Trade 24/7)", evaluate_logic(False, 'SHORT', u, c, h, l, bull, bear, m))

if __name__ == '__main__':
    main()
