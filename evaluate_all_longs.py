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
    print(f"Evaluating LONG models on {device}...")
    
    # Load past 4000 candles (approx 40 days)
    df = pd.read_csv('data_storage/BTC_USDT_15m_processed.csv')
    df = df.iloc[-4000:].copy() 
    
    feature_cols = get_feature_cols()
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    features_np = df[feature_cols].values.astype(np.float32)
    
    model_paths = glob.glob('models/trial_long_*.pth')
    model_paths.sort(key=os.path.getmtime, reverse=True)
    
    print(f"Found {len(model_paths)} LONG models to evaluate.")
    
    results = []
    
    for path in model_paths:
        config_path = path.replace('.pth', '_config.json')
        if not os.path.exists(config_path):
            continue
            
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            
        s_len = cfg.get('seq_len', 128)
        LONG_TP = cfg.get('take_profit', 0.05)
        LONG_SL = cfg.get('stop_loss', 0.015)
        LONG_MAX_BARS = cfg.get('max_hold_bars', 24)
        
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
        all_bull = []
        
        with torch.no_grad():
            for start in range(0, len(features_np) - s_len + 1, batch_size):
                end = min(start + batch_size, len(features_np) - s_len + 1)
                batch = [features_np[i : i + s_len] for i in range(start, end)]
                tensor_batch = torch.tensor(np.array(batch)).to(device)
                
                out = model(tensor_batch)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                all_bull.extend(probs)
                
        # Trading simulation
        position = 'flat'
        entry_price = 0
        bars_held = 0
        trades = 0
        wins = 0
        total_pnl = 0
        
        for idx in range(len(all_bull)):
            prob_bull = all_bull[idx]
            bar_idx = s_len - 1 + idx
            
            if bar_idx < len(close_prices):
                close_price = close_prices[bar_idx]
                high_price = high_prices[bar_idx]
                low_price = low_prices[bar_idx]
                
                if position == 'flat':
                    if prob_bull >= 0.50:
                        position = 'long'
                        entry_price = close_price
                        bars_held = 0
                else:
                    bars_held += 1
                    closed = False
                    pnl = 0
                    
                    tp_price = entry_price * (1 + LONG_TP)
                    sl_price = entry_price * (1 - LONG_SL)
                    
                    if low_price <= sl_price:
                        pnl = (sl_price - entry_price) / entry_price * 100
                        closed = True
                    elif high_price >= tp_price:
                        pnl = (tp_price - entry_price) / entry_price * 100
                        closed = True
                    elif bars_held >= LONG_MAX_BARS:
                        pnl = (close_price - entry_price) / entry_price * 100
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
            'TP%': f"{LONG_TP*100:.1f}%",
            'SL%': f"{LONG_SL*100:.1f}%",
            'MaxHold': LONG_MAX_BARS,
            'Trades': trades,
            'Win Rate': win_rate,
            'Total PnL': total_pnl,
            'Avg PnL': total_pnl / trades if trades > 0 else 0
        })
        
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res[df_res['Trades'] >= 5]
        df_res = df_res.sort_values('Total PnL', ascending=False).head(15)
        
        pd.set_option('display.float_format', '{:.2f}'.format)
        print("\nTop Long Models on Recent History (4000 bars):")
        print(df_res.to_string(index=False))

if __name__ == "__main__":
    evaluate_models()
