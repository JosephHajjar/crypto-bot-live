import os
import glob
import json
import torch
import numpy as np
import pandas as pd
import sys

from ml.model import AttentionLSTMModel
from data.feature_engineer import get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_models():
    print(f"Evaluating short models on {device}...")
    
    # Load past 2000 candles (approx 20 days)
    df = pd.read_csv('data_storage/BTC_USDT_15m_short_processed.csv')
    df = df.iloc[-4000:].copy() # Enough data to support seq_len and provide good backtest
    
    feature_cols = get_feature_cols()    77.8%	+7.08%	0.39%
    All Time	199	85.4%	+217.77%	1.09%
    
    close_prices = df['close'].values
    high_prices = df['high'].values
    features_np = df[feature_cols].values.astype(np.float32)
    unix_times = (pd.to_datetime(df['timestamp']).astype(np.int64) // 10**9).values
    
    SHORT_TP = 0.015
    SHORT_SL = 0.008
    SHORT_MAX_BARS = 8
    
    model_paths = glob.glob('models_short/trial_*.pth')
    model_paths.sort(key=os.path.getmtime, reverse=True)
    model_paths = model_paths[:25]
    if 'models_short/holy_grail_short.pth' in glob.glob('models_short/*.pth'):
        model_paths.append('models_short\\holy_grail_short.pth')
        
    print(f"Found {len(model_paths)} models to evaluate.")
    
    results = []
    
    for path in model_paths:
        config_path = path.replace('.pth', '_config.json')
        if not os.path.exists(config_path):
            continue
            
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            
        s_short = cfg.get('seq_len', 128)
        
        try:
            model = AttentionLSTMModel(
                input_dim=cfg['input_dim'], hidden_dim=cfg['hidden_dim'],
                num_layers=cfg['num_layers'], output_dim=2,
                dropout=cfg['dropout'], num_heads=cfg['num_heads']
            ).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
        except Exception:
            continue
            
        # Inference
        batch_size = 64
        all_bear = []
        
        with torch.no_grad():
            for start in range(0, len(features_np) - s_short + 1, batch_size):
                end = min(start + batch_size, len(features_np) - s_short + 1)
                batch = [features_np[i : i + s_short] for i in range(start, end)]
                tensor_batch = torch.tensor(np.array(batch)).to(device)
                
                out = model(tensor_batch)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                all_bear.extend(probs)
                
        # Trading simulation
        position = 'flat'
        entry_price = 0
        bars_held = 0
        trades = 0
        wins = 0
        total_pnl = 0
        
        for idx in range(len(all_bear)):
            prob_bear = all_bear[idx]
            bar_idx = s_short - 1 + idx
            
            if bar_idx < len(close_prices):
                close_price = close_prices[bar_idx]
                high_price = high_prices[bar_idx]
                
                if position == 'flat':
                    if prob_bear >= 0.50:
                        position = 'short'
                        entry_price = close_price
                        bars_held = 0
                else:
                    bars_held += 1
                    closed = False
                    pnl = 0
                    
                    tp_price = entry_price * (1 - SHORT_TP)
                    sl_price = entry_price * (1 + SHORT_SL)
                    
                    if high_price >= sl_price:
                        pnl = (entry_price - sl_price) / entry_price * 100
                        closed = True
                    elif close_price <= tp_price:
                        pnl = (entry_price - tp_price) / entry_price * 100
                        closed = True
                    elif bars_held >= SHORT_MAX_BARS:
                        pnl = (entry_price - close_price) / entry_price * 100
                        closed = True
                        
                    if closed:
                        trades += 1
                        total_pnl += pnl
                        if pnl > 0:
                            wins += 1
                        position = 'flat'
                        
        win_rate = (wins / trades * 100) if trades > 0 else 0
        print(f"Eval {os.path.basename(path)} -> Trades: {trades}, PnL: {total_pnl:.2f}%, WinRate: {win_rate:.1f}%")
        results.append({
            'Model': os.path.basename(path),
            'Trades': trades,
            'Win Rate': win_rate,
            'Total PnL': total_pnl,
            'Avg PnL': total_pnl / trades if trades > 0 else 0
        })
        
    df_res = pd.DataFrame(results)
    # Filter out models with < 10 trades to avoid 100% win rate on 1 trade anomalies
    df_res = df_res[df_res['Trades'] >= 10]
    df_res = df_res.sort_values('Total PnL', ascending=False).head(15)
    
    pd.set_option('display.float_format', '{:.2f}'.format)
    print("\nTop 15 Short Models on Recent History (4000 bars):")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    evaluate_models()
