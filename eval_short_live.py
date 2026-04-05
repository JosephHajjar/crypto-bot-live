import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ml.model import AttentionLSTMModel
from data.feature_engineer_btc import get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def precompute_signals():
    df = pd.read_csv('data_storage/BTC_USDT_15m_processed.csv')
    df = df.copy() 
    
    with open('models_short/holy_grail_short_config.json', 'r') as f:
        cfg_short = json.load(f)
    s_short = cfg_short.get('seq_len', 128)
    
    model_short = AttentionLSTMModel(
        input_dim=cfg_short['input_dim'], hidden_dim=cfg_short['hidden_dim'],
        num_layers=cfg_short['num_layers'], output_dim=2, dropout=cfg_short['dropout'], num_heads=cfg_short['num_heads']
    ).to(device)
    model_short.load_state_dict(torch.load('models_short/holy_grail_short.pth', map_location=device, weights_only=True))
    model_short.eval()

    feature_cols = get_feature_cols()
    
    unix_times = df['timestamp'].values
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    features_np = df[feature_cols].values.astype(np.float32)
    
    batch_size = 256
    all_bear = []
    
    with torch.no_grad():
        for start in range(0, len(features_np) - s_short + 1, batch_size):
            end = min(start + batch_size, len(features_np) - s_short + 1)
            batch_s = [features_np[i : i + s_short] for i in range(start, end)]
            t_short = torch.tensor(np.array(batch_s)).to(device)
            p_bear = torch.softmax(model_short(t_short), dim=1)[:, 1].cpu().numpy().tolist()
            all_bear.extend(p_bear)
            
    return unix_times, close_prices, high_prices, low_prices, all_bear, s_short

def evaluate_short_logic(logic_type, times, close_prices, high_prices, low_prices, all_bear, max_seq):
    position = 'flat'
    entry_price = 0
    bars_held = 0
    trades = []
    
    for idx in range(len(all_bear)):
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
                if prob_bear >= 0.50 and can_trade:
                    position = 'short'
                    entry_price = c_p
                    bars_held = 0
            else:
                bars_held += 1
                closed = False
                pnl = 0
                
                if logic_type == "SHORT_SCALPER":
                    # 1.5% TP, 0.8% SL, 8 bars
                    tp_p = entry_price * 0.985
                    sl_p = entry_price * 1.008
                    if h_p >= sl_p:
                        pnl = (entry_price - sl_p)/entry_price*100; closed=True
                    elif l_p <= tp_p:
                        pnl = (entry_price - tp_p)/entry_price*100; closed=True
                    elif bars_held >= 8:
                        pnl = (entry_price - c_p)/entry_price*100; closed=True
                elif logic_type == "SHORT_OPTUNA":
                    # Optuna: 4.5% TP, 5.0% SL, 4 bars
                    tp_p = entry_price * 0.955
                    sl_p = entry_price * 1.050
                    if h_p >= sl_p:
                        pnl = (entry_price - sl_p)/entry_price*100; closed=True
                    elif l_p <= tp_p:
                        pnl = (entry_price - tp_p)/entry_price*100; closed=True
                    elif bars_held >= 4:
                        pnl = (entry_price - c_p)/entry_price*100; closed=True
                elif logic_type == "CURRENT_LIVE":
                    # Live JSON Config: 1.75% TP, 3.75% SL, 12 bars
                    tp_p = entry_price * (1 - 0.0175)
                    sl_p = entry_price * (1 + 0.0375)
                    if h_p >= sl_p:
                        pnl = (entry_price - sl_p)/entry_price*100; closed=True
                    elif l_p <= tp_p:
                        pnl = (entry_price - tp_p)/entry_price*100; closed=True
                    elif bars_held >= 12:
                        pnl = (entry_price - c_p)/entry_price*100; closed=True
                        
                if closed:
                    trades.append({'time': dt, 'pnl': pnl})
                    position = 'flat'
                    
    return trades

def print_stats(name, trades):
    if not trades: return
    df_trades = pd.DataFrame(trades)
    
    total_pnl = df_trades['pnl'].sum()
    win_rate = (df_trades['pnl'] > 0).mean() * 100
    
    last_time = df_trades['time'].iloc[-1]
    
    def calculate_period(days):
        cutoff = last_time - timedelta(days=days)
        period_df = df_trades[df_trades['time'] >= cutoff]
        if period_df.empty: return 0, 0, 0
        return len(period_df), (period_df['pnl'] > 0).mean() * 100, period_df['pnl'].sum()
    
    d30_trades, d30_wr, d30_pnl = calculate_period(30)
    d60_trades, d60_wr, d60_pnl = calculate_period(60)
    d90_trades, d90_wr, d90_pnl = calculate_period(90)
    
    print(f"\n[{name}]")
    print(f"  30 Days : {d30_trades:4d} Trades | Win: {d30_wr:5.1f}% | ROI: {d30_pnl:6.2f}%")
    print(f"  60 Days : {d60_trades:4d} Trades | Win: {d60_wr:5.1f}% | ROI: {d60_pnl:6.2f}%")
    print(f"  90 Days : {d90_trades:4d} Trades | Win: {d90_wr:5.1f}% | ROI: {d90_pnl:6.2f}%")
    print(f"  All Time: {len(trades):4d} Trades | Win: {win_rate:5.1f}% | ROI: {total_pnl:6.2f}%")

def main():
    u, c, h, l, bear, m = precompute_signals()
    
    t1 = evaluate_short_logic("SHORT_SCALPER", u, c, h, l, bear, m)
    t2 = evaluate_short_logic("SHORT_OPTUNA", u, c, h, l, bear, m)
    t3 = evaluate_short_logic("CURRENT_LIVE", u, c, h, l, bear, m)
    
    print("\n================ FINAL SHORT EVALUATION ================")
    print_stats("1. SURGICAL SCALPER (1.50% TP, 0.8% SL, 8 Bar Hold)", t1)
    print_stats("2. NEW OPTUNA RANGE (4.50% TP, 5.0% SL, 4 Bar Hold)", t2)
    print_stats("3. CURRENT LIVE JSON (1.75% TP, 3.75% SL, 12 Bar Hold)", t3)

if __name__ == '__main__':
    main()
