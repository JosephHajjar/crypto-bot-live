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
    # Load 10000 candles to give us ~104 days of history
    df = pd.read_csv('data_storage/BTC_USDT_15m_processed.csv')
    df = df.iloc[-10000:].copy() 
    
    with open('models/holy_grail_config.json', 'r') as f:
        cfg_long = json.load(f)
    s_long = cfg_long.get('seq_len', 128)
    
    model_long = AttentionLSTMModel(
        input_dim=cfg_long['input_dim'], hidden_dim=cfg_long['hidden_dim'],
        num_layers=cfg_long['num_layers'], output_dim=2, dropout=cfg_long['dropout'], num_heads=cfg_long['num_heads']
    ).to(device)
    model_long.load_state_dict(torch.load('models/holy_grail.pth', map_location=device, weights_only=True))
    model_long.eval()

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
            
    return unix_times, close_prices, high_prices, low_prices, all_bull, all_bear, MAX_SEQ_LEN, cfg_long, cfg_short

def evaluate_strategies_for_period(times, close_prices, high_prices, low_prices, all_bull, all_bear, max_seq, cfg_l, cfg_s, days=None):
    if days is not None:
        last_time = times[-1]
        if isinstance(last_time, str):
            last_dt = pd.to_datetime(last_time)
        else:
            last_dt = datetime.utcfromtimestamp(last_time)
        cutoff_dt = last_dt - timedelta(days=days)
    else:
        cutoff_dt = pd.to_datetime('1970-01-01')

    # PROPORTIONAL STRATEGY
    position_prop = None
    entry_price_prop = 0
    trades_prop = []
    
    # THRESHOLD STRATEGY
    position_thresh = 'flat'
    entry_price_thresh = 0
    bars_held_thresh = 0
    trades_thresh = []
    
    LONG_TP = cfg_l.get('take_profit', 0.0125)
    LONG_SL = cfg_l.get('stop_loss', 0.025)
    LONG_MAX_BARS = cfg_l.get('max_hold_bars', 12)
    SHORT_TP = cfg_s.get('take_profit', 0.015)
    SHORT_SL = cfg_s.get('stop_loss', 0.008)
    SHORT_MAX_BARS = cfg_s.get('max_hold_bars', 8)
    
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
                
            if dt < cutoff_dt:
                continue

            c_p = close_prices[bar_idx]
            h_p = high_prices[bar_idx]
            l_p = low_prices[bar_idx]
            
            # 1. Proportional Logic
            target_position = 'long' if prob_bull > prob_bear else 'short'
            
            if position_prop is None:
                position_prop = target_position
                entry_price_prop = c_p
            elif position_prop != target_position:
                if position_prop == 'long':
                    pnl_prop = (c_p - entry_price_prop) / entry_price_prop * 100
                else:
                    pnl_prop = (entry_price_prop - c_p) / entry_price_prop * 100
                    
                trades_prop.append({'time': dt, 'pnl': pnl_prop, 'type': position_prop})
                position_prop = target_position
                entry_price_prop = c_p
                
            # 2. Threshold Logic
            if position_thresh == 'flat':
                if prob_bull >= 0.60:
                    position_thresh = 'long'
                    entry_price_thresh = c_p
                    bars_held_thresh = 0
                elif prob_bear >= 0.50:
                    position_thresh = 'short'
                    entry_price_thresh = c_p
                    bars_held_thresh = 0
            else:
                bars_held_thresh += 1
                closed = False
                pnl = 0
                if position_thresh == 'long':
                    tp_price = entry_price_thresh * (1 + LONG_TP)
                    sl_price = entry_price_thresh * (1 - LONG_SL)
                    if l_p <= sl_price:
                        pnl = (sl_price - entry_price_thresh) / entry_price_thresh * 100
                        closed = True
                    elif h_p >= tp_price:
                        pnl = (tp_price - entry_price_thresh) / entry_price_thresh * 100
                        closed = True
                    elif bars_held_thresh >= LONG_MAX_BARS:
                        pnl = (c_p - entry_price_thresh) / entry_price_thresh * 100
                        closed = True
                elif position_thresh == 'short':
                    tp_price = entry_price_thresh * (1 - SHORT_TP)
                    sl_price = entry_price_thresh * (1 + SHORT_SL)
                    if h_p >= sl_price:
                        pnl = (entry_price_thresh - sl_price) / entry_price_thresh * 100
                        closed = True
                    elif l_p <= tp_price:
                        pnl = (entry_price_thresh - tp_price) / entry_price_thresh * 100
                        closed = True
                    elif bars_held_thresh >= SHORT_MAX_BARS:
                        pnl = (entry_price_thresh - c_p) / entry_price_thresh * 100
                        closed = True
                        
                if closed:
                    trades_thresh.append({'time': dt, 'pnl': pnl, 'type': position_thresh})
                    position_thresh = 'flat'
                    entry_price_thresh = 0
                    
    return trades_prop, trades_thresh

def get_stats(trades):
    if not trades: return 0, 0, 0, 0, 0
    df_trades = pd.DataFrame(trades)
    total_pnl = df_trades['pnl'].sum()
    win_rate = (df_trades['pnl'] > 0).mean() * 100
    fees = 0.07 * len(trades)
    net_roi = total_pnl - fees
    cumulative_pnl = df_trades['pnl'].cumsum()
    max_drawdown = (cumulative_pnl.cummax() - cumulative_pnl).max()
    return len(trades), win_rate, total_pnl, net_roi, max_drawdown

def main():
    u, c, h, l, bull, bear, m, cfg_l, cfg_s = load_data_and_predict()
    
    periods = [7, 14, 30, 60, 90, 100]
    
    print(f"\n==============================================")
    print(f" ROI COMPARISON: PROPORTIONAL VS THRESHOLD BOT ")
    print(f"==============================================\n")
    
    results = []
    
    for days in periods:
        trades_prop, trades_thresh = evaluate_strategies_for_period(u, c, h, l, bull, bear, m, cfg_l, cfg_s, days=days)
        
        p_c, p_wr, p_raw, p_net, p_md = get_stats(trades_prop)
        t_c, t_wr, t_raw, t_net, t_md = get_stats(trades_thresh)
        
        results.append({
            'Days': days if days < 100 else 'All (~100d)',
            'Prop Net ROI': p_net,
            'Prop Win%': p_wr,
            'Prop Trades': p_c,
            'Prop MDD': p_md,
            'Thresh Net ROI': t_net,
            'Thresh Win%': t_wr,
            'Thresh Trades': t_c,
            'Thresh MDD': t_md,
        })
        
    df_res = pd.DataFrame(results)
    
    # Custom print
    for idx, row in df_res.iterrows():
        print(f"--- LAST {row['Days']} DAYS ---")
        print(f"PROPORTIONAL (Continuous) : Net ROI: {row['Prop Net ROI']:+6.2f}% | Win Rate: {row['Prop Win%']:5.1f}% | Trades: {row['Prop Trades']:4d} | Max DD: {row['Prop MDD']:5.2f}%")
        print(f"THRESHOLD    (Selective)  : Net ROI: {row['Thresh Net ROI']:+6.2f}% | Win Rate: {row['Thresh Win%']:5.1f}% | Trades: {row['Thresh Trades']:4d} | Max DD: {row['Thresh MDD']:5.2f}%")
        diff = row['Prop Net ROI'] - row['Thresh Net ROI']
        winner = "PROPORTIONAL" if diff > 0 else "THRESHOLD"
        print(f"WINNER: {winner} (by {abs(diff):.2f}% margin)\n")

if __name__ == '__main__':
    main()
