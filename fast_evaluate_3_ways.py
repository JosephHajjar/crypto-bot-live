import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ml.model import AttentionLSTMModel
from data.feature_engineer_btc import get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_and_predict():
    print("Loading data and predicting signals...")
    df = pd.read_csv('data_storage/BTC_USDT_15m_processed.csv')
    df = df.iloc[-4000:].copy() # Using past ~40 days explicitly for this 1D/7D/14D/30D split emulation
    
    # LOAD LONG MODEL (255)
    with open('models/trial_255_config.json', 'r') as f:
        cfg_long = json.load(f)
    s_long = cfg_long.get('seq_len', 128)
    
    model_long = AttentionLSTMModel(
        input_dim=cfg_long['input_dim'], hidden_dim=cfg_long['hidden_dim'],
        num_layers=cfg_long['num_layers'], output_dim=2, dropout=cfg_long['dropout'], num_heads=cfg_long['num_heads']
    ).to(device)
    model_long.load_state_dict(torch.load('models/trial_255.pth', map_location=device, weights_only=True))
    model_long.eval()

    # LOAD SHORT MODEL (270)
    with open('models_short/holy_grail_short_config.json', 'r') as f:
        cfg_short = json.load(f)
    s_short = cfg_short.get('seq_len', 128)
    
    model_short = AttentionLSTMModel(
        input_dim=cfg_short['input_dim'], hidden_dim=cfg_short['hidden_dim'],
        num_layers=cfg_short['num_layers'], output_dim=2, dropout=cfg_short['dropout'], num_heads=cfg_short['num_heads']
    ).to(device)
    model_short.load_state_dict(torch.load('models_short/holy_grail_short.pth', map_location=device, weights_only=True))
    model_short.eval()

    MAX_SEQ_LEN = max(s_long, s_short)
    feature_cols = get_feature_cols()
    
    unix_times = df['timestamp'].values
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    features_np = df[feature_cols].values.astype(np.float32)
    
    batch_size = 256
    all_bull = []
    all_bear = []
    
    with torch.no_grad():
        for start in range(0, len(features_np) - MAX_SEQ_LEN + 1, batch_size):
            end = min(start + batch_size, len(features_np) - MAX_SEQ_LEN + 1)
            
            batch_l = [features_np[i + MAX_SEQ_LEN - s_long : i + MAX_SEQ_LEN] for i in range(start, end)]
            batch_s = [features_np[i + MAX_SEQ_LEN - s_short : i + MAX_SEQ_LEN] for i in range(start, end)]
            
            t_long = torch.tensor(np.array(batch_l)).to(device)
            t_short = torch.tensor(np.array(batch_s)).to(device)
            
            p_bull = torch.softmax(model_long(t_long), dim=1)[:, 1].cpu().numpy().tolist()
            p_bear = torch.softmax(model_short(t_short), dim=1)[:, 1].cpu().numpy().tolist()
            
            all_bull.extend(p_bull)
            all_bear.extend(p_bear)
            
    return unix_times, close_prices, high_prices, low_prices, all_bull, all_bear, MAX_SEQ_LEN

def evaluate_logic(logic_type, times, close_prices, high_prices, low_prices, all_bull, all_bear, max_seq):
    position = 'flat'
    entry_price = 0
    bars_held = 0
    
    trades = []
    highest_p = 0
    tsl_sl = 0
    tsl_active = False
    
    for idx in range(len(all_bull)):
        prob_bull = all_bull[idx]
        prob_bear = all_bear[idx]
        bar_idx = max_seq - 1 + idx
        
        if bar_idx < len(times):
            time_val = times[bar_idx]
            if isinstance(time_val, str):
                dt = pd.to_datetime(time_val)
            else:
                dt = datetime.utcfromtimestamp(time_val)
            utc_hour = dt.hour
            can_trade = not (13 <= utc_hour <= 16)
            
            c_p = close_prices[bar_idx]
            h_p = high_prices[bar_idx]
            l_p = low_prices[bar_idx]
            
            if position == 'flat':
                if prob_bull >= 0.60 and prob_bear >= 0.50 and can_trade:
                    pass
                elif prob_bull >= 0.60 and can_trade:
                    position = 'long'
                    entry_price = c_p
                    bars_held = 0
                    highest_p = h_p
                    tsl_active = False
                    tsl_sl = entry_price * (1 - 0.025) # Initial SL for TSL 2.5%
                elif prob_bear >= 0.50 and can_trade:
                    position = 'short'
                    entry_price = c_p
                    bars_held = 0
            else:
                bars_held += 1
                closed = False
                pnl = 0
                
                if position == 'long':
                    if logic_type == "ORIGINAL":
                        # Original bounds: 1.25% TP, 2.5% SL, 12 bars
                        tp_p = entry_price * 1.0125
                        sl_p = entry_price * 0.975
                        if l_p <= sl_p:
                            pnl = (sl_p - entry_price)/entry_price*100; closed=True
                        elif h_p >= tp_p:
                            pnl = (tp_p - entry_price)/entry_price*100; closed=True
                        elif bars_held >= 12:
                            pnl = (c_p - entry_price)/entry_price*100; closed=True
                            
                    elif logic_type == "STATIC":
                        # New static optuna: 7.5% TP, 1.5% SL, 8 bars
                        tp_p = entry_price * 1.075
                        sl_p = entry_price * 0.985
                        if l_p <= sl_p:
                            pnl = (sl_p - entry_price)/entry_price*100; closed=True
                        elif h_p >= tp_p:
                            pnl = (tp_p - entry_price)/entry_price*100; closed=True
                        elif bars_held >= 8:
                            pnl = (c_p - entry_price)/entry_price*100; closed=True
                            
                    elif logic_type == "TRAILING":
                        # Trailing: init_sl=2.5%, act=0.5%, trail=0.5%, 16 bars
                        if h_p > highest_p: highest_p = h_p
                        
                        if highest_p >= entry_price * 1.005:
                            tsl_active = True
                            
                        if tsl_active:
                            prop_sl = highest_p * 0.995
                            if prop_sl > tsl_sl: tsl_sl = prop_sl
                            
                        if l_p <= tsl_sl:
                            # Assume slippage out at exactly the stop loss marker
                            pnl = (tsl_sl - entry_price)/entry_price*100; closed=True
                        elif bars_held >= 16:
                            pnl = (c_p - entry_price)/entry_price*100; closed=True
                            
                elif position == 'short':
                    # Global Short Logic (Trial 270: 1.5% TP, 0.8% SL, 8 bars)
                    tp_p = entry_price * 0.985
                    sl_p = entry_price * 1.008
                    if h_p >= sl_p:
                        pnl = (entry_price - sl_p)/entry_price*100; closed=True
                    elif l_p <= tp_p:
                        pnl = (entry_price - tp_p)/entry_price*100; closed=True
                    elif bars_held >= 8:
                        pnl = (entry_price - c_p)/entry_price*100; closed=True
                        
                if closed:
                    trades.append({'time': dt, 'pnl': pnl})
                    position = 'flat'
                    
    return trades

def print_stats(name, trades):
    # Analyze exactly over the last 40 days chunks
    if not trades: return
    
    last_dt = trades[-1]['time']
    if isinstance(last_dt, str):
        last_dt = pd.to_datetime(last_dt)
    df_trades = pd.DataFrame(trades)
    
    if df_trades.empty: return
    
    total_pnl = df_trades['pnl'].sum()
    win_rate = (df_trades['pnl'] > 0).mean() * 100
    
    print(f"[{name}] -> {len(trades)} Trades | Win: {win_rate:.1f}% | ROI: {total_pnl:.2f}%")

def main():
    u, c, h, l, bull, bear, m = load_data_and_predict()
    
    print("\nEvaluating Original (1.25% TP, 2.5% SL)...")
    t1 = evaluate_logic("ORIGINAL", u, c, h, l, bull, bear, m)
    
    print("Evaluating Static Optimized (7.5% TP, 1.5% SL)...")
    t2 = evaluate_logic("STATIC", u, c, h, l, bull, bear, m)
    
    print("Evaluating Trailing Stop Loss (0.5% Act, 0.5% Pad, 2.5% Init SL)...")
    t3 = evaluate_logic("TRAILING", u, c, h, l, bull, bear, m)
    
    print("\n--- SUMMARY OF RECENT (40 DAYS) ---")
    print_stats("1. Original T-255  ", t1)
    print_stats("2. Static Optimized", t2)
    print_stats("3. Trailing T-255  ", t3)

if __name__ == '__main__':
    main()
