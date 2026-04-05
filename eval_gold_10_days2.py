import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ml.model import AttentionLSTMModel
from data.feature_engineer import dynamic_features_and_labels, get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_gold():
    print("Loading engineered PAXG static data...")
    df_static = pd.read_csv('data_storage/PAXG_USDT_15m_static.csv')
    
    print("Applying dynamic labels...")
    # Just generic labels since we only need the features
    df = dynamic_features_and_labels(df_static, 14, 14, 8, 1.5)
    
    # Filter strictly to the last 10 days
    cutoff_time = int((datetime.now() - timedelta(days=10)).timestamp() * 1000)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df_eval = df[df['timestamp'] >= cutoff_time].copy()
    print(f"Data ready. Evaluating {len(df_eval)} recent candles (10 days)")
    
    with open('models_gold_long/holy_grail_config.json', 'r') as f:
        cfg = json.load(f)
    seq_len = cfg.get('seq_len', 128)
    
    model = AttentionLSTMModel(
        input_dim=cfg['input_dim'], hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'], output_dim=2, dropout=cfg['dropout'], num_heads=cfg['num_heads']
    ).to(device)
    model.load_state_dict(torch.load('models_gold_long/holy_grail.pth', map_location=device, weights_only=True))
    model.eval()

    feature_cols = get_feature_cols()
    
    # Needs sequence context, so we read from df instead of df_eval for the features array
    unix_times = df['timestamp'].values
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    features_np = df[feature_cols].values.astype(np.float32)

    df_target_idx = df_eval.index.min()
    start_idx = max(seq_len, df_target_idx if pd.notna(df_target_idx) else seq_len)
    
    position = 'flat'
    entry_price = 0
    bars_held = 0
    trades = []
    
    print("Simulating trades over the last 10 days...")
    with torch.no_grad():
        for i in range(start_idx, len(df)):
            seq = features_np[i - seq_len : i]
            t_seq = torch.tensor(np.array([seq])).to(device)
            
            prob_bull = torch.softmax(model(t_seq), dim=1)[0][1].item()
            
            time_val = unix_times[i]
            dt = datetime.utcfromtimestamp(time_val / 1000) if time_val > 1e12 else datetime.utcfromtimestamp(time_val)
            
            c_p = close_prices[i]
            h_p = high_prices[i]
            l_p = low_prices[i]
            
            if position == 'flat':
                if prob_bull >= 0.60:
                    position = 'long'
                    entry_price = c_p
                    bars_held = 0
            else:
                bars_held += 1
                closed = False
                pnl = 0
                
                # Standard Gold Evaluation TP/SL: 0.25% TP, 0.4% SL (since volatility is low)
                tp_p = entry_price * 1.0025
                sl_p = entry_price * 0.996
                
                if h_p >= tp_p:
                    pnl = (tp_p - entry_price)/entry_price * 100
                    closed = True
                elif l_p <= sl_p:
                    pnl = (sl_p - entry_price)/entry_price * 100
                    closed = True
                elif bars_held >= 12:
                    pnl = (c_p - entry_price)/entry_price * 100
                    closed = True
                    
                if closed:
                    trades.append({'time': dt, 'pnl': pnl})
                    position = 'flat'
                    
    print("\n========= GOLD MODEL (PAXG) - 10 DAY ROI =========")
    if not trades:
        print("0 Trades Executed in the last 10 days! (Market Flat/Unsafe)")
    else:
        df_trades = pd.DataFrame(trades)
        total_pnl = df_trades['pnl'].sum()
        win_rate = (df_trades['pnl'] > 0).mean() * 100
        print(f"Total Trades : {len(trades)}")
        print(f"Win Rate     : {win_rate:.1f}%")
        print(f"Total ROI    : {total_pnl:.2f}%")

if __name__ == '__main__':
    evaluate_gold()
