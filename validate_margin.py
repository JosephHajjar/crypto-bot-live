import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from ml.model import AttentionLSTMModel
from data.feature_engineer_btc import get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_and_predict():
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
            all_bull.extend(torch.softmax(model_long(t_long), dim=1)[:, 1].cpu().numpy().tolist())
            all_bear.extend(torch.softmax(model_short(t_short), dim=1)[:, 1].cpu().numpy().tolist())
            
    return unix_times, close_prices, all_bull, all_bear, MAX_SEQ_LEN

def evaluate_strategy(times, close_prices, all_bull, all_bear, max_seq, use_entry_margin=True):
    position = None
    entry_price = 0
    trades = []
    
    ENTRY_MARGIN = 0.105 if use_entry_margin else 0.0008
    FLIP_MARGIN = 0.0008
    
    for idx in range(len(all_bull)):
        prob_bull = all_bull[idx]
        prob_bear = all_bear[idx]
        bar_idx = max_seq - 1 + idx
        
        if bar_idx < len(times):
            time_val = times[bar_idx]
            dt = datetime.utcfromtimestamp(time_val) if isinstance(time_val, (int, float)) else pd.to_datetime(time_val)
            c_p = close_prices[bar_idx]
            
            diff = abs(prob_bull - prob_bear)
            
            if position is None:
                if diff > ENTRY_MARGIN:
                    position = 'long' if prob_bull > prob_bear else 'short'
                    entry_price = c_p
            else:
                if diff > FLIP_MARGIN:
                    target_position = 'long' if prob_bull > prob_bear else 'short'
                    if target_position != position:
                        pnl = (c_p - entry_price) / entry_price * 100 if position == 'long' else (entry_price - c_p) / entry_price * 100
                        trades.append({'time': dt, 'pnl': pnl, 'type': position, 'entry': entry_price, 'exit': c_p})
                        position = target_position
                        entry_price = c_p
                        
    return trades

def print_stats(name, trades):
    if not trades: 
        print(f"\n--- {name} RESULTS ---\nNo trades executed.")
        return
    
    df_trades = pd.DataFrame(trades)
    total_pnl = df_trades['pnl'].sum()
    win_rate = (df_trades['pnl'] > 0).mean() * 100
    fees = 0.07 * len(trades)
    pnl_with_fees = total_pnl - fees
    
    print(f"\n--- {name} RESULTS ---")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate:     {win_rate:.1f}%")
    print(f"Raw ROI:      {total_pnl:.2f}%")
    print(f"Fee (-0.07%): -{fees:.2f}%")
    print(f"Net ROI:      {pnl_with_fees:.2f}%")

def main():
    u, c, bull, bear, m = load_data_and_predict()
    t1 = evaluate_strategy(u, c, bull, bear, m, use_entry_margin=True)
    print_stats("WITH Initial Margin Seeding (0.105)", t1)
    
    t2 = evaluate_strategy(u, c, bull, bear, m, use_entry_margin=False)
    print_stats("WITHOUT Initial Seeding (Jumps right in @ 0.0008)", t2)

if __name__ == '__main__':
    main()
