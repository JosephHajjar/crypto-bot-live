import os
import json
import torch
import numpy as np
import pandas as pd
import optuna
from datetime import datetime

from ml.model import AttentionLSTMModel
from data.feature_engineer import get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def precompute_signals():
    print("Precomputing signals for SHORT holy_grail_short (trial_270)...")
    df = pd.read_csv('data_storage/BTC_USDT_15m_processed.csv')
    df = df.iloc[-8000:].copy() # Test on recent 8000 bars
    
    with open('models_short/holy_grail_short_config.json', 'r') as f:
        cfg = json.load(f)
        
    s_len = cfg.get('seq_len', 128)
    feature_cols = get_feature_cols()
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    features_np = df[feature_cols].values.astype(np.float32)
    
    model = AttentionLSTMModel(
        input_dim=cfg['input_dim'], hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'], output_dim=2, dropout=cfg['dropout'], num_heads=cfg['num_heads']
    ).to(device)
    model.load_state_dict(torch.load('models_short/holy_grail_short.pth', map_location=device, weights_only=True))
    model.eval()
    
    batch_size = 256
    all_probs = []
    
    with torch.no_grad():
        for start in range(0, len(features_np) - s_len + 1, batch_size):
            end = min(start + batch_size, len(features_np) - s_len + 1)
            batch = [features_np[i : i + s_len] for i in range(start, end)]
            tensor_batch = torch.tensor(np.array(batch)).to(device)
            out = model(tensor_batch)
            # PROBABILITY OF BEARISH OUTCOME
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy().tolist()
            all_probs.extend(probs)
            
    print(f"Precomputed {len(all_probs)} SHORT signals.")
    return df, close_prices, high_prices, low_prices, all_probs, s_len

class FastShortObjective:
    def __init__(self, close_prices, high_prices, low_prices, all_probs, s_len):
        self.close_prices = close_prices
        self.high_prices = high_prices
        self.low_prices = low_prices
        self.all_probs = np.array(all_probs)
        self.s_len = s_len
        
    def __call__(self, trial):
        # We allow it to test massive ranges for TP and SL to find anomalies!
        SHORT_TP = trial.suggest_float('take_profit', 0.01, 0.15, step=0.005)
        SHORT_SL = trial.suggest_float('stop_loss', 0.005, 0.05, step=0.005)
        SHORT_MAX_BARS = trial.suggest_int('max_hold_bars', 4, 48, step=4)
        
        position = 'flat'
        entry_price = 0
        bars_held = 0
        trades = 0
        total_pnl = 0
        wins = 0
        
        for idx in range(len(self.all_probs)):
            prob_bear = self.all_probs[idx]
            bar_idx = self.s_len - 1 + idx
            
            if bar_idx < len(self.close_prices):
                close_price = self.close_prices[bar_idx]
                high_price = self.high_prices[bar_idx]
                low_price = self.low_prices[bar_idx]
                
                if position == 'flat':
                    if prob_bear >= 0.50:
                        position = 'short'
                        entry_price = close_price
                        bars_held = 0
                else:
                    bars_held += 1
                    closed = False
                    pnl = 0
                    
                    # For a SHORT, TP is lower than entry, SL is higher than entry
                    tp_price = entry_price * (1 - SHORT_TP)
                    sl_price = entry_price * (1 + SHORT_SL)
                    
                    # Did price hit Stop Loss (price went too high)?
                    if high_price >= sl_price:
                        pnl = (entry_price - sl_price) / entry_price * 100
                        closed = True
                    # Did price hit Take Profit (price dropped low enough)?
                    elif low_price <= tp_price:
                        pnl = (entry_price - tp_price) / entry_price * 100
                        closed = True
                    # Max duration reached
                    elif bars_held >= SHORT_MAX_BARS:
                        pnl = (entry_price - close_price) / entry_price * 100
                        closed = True
                        
                    if closed:
                        total_pnl += pnl
                        trades += 1
                        if pnl > 0: wins += 1
                        position = 'flat'
                        
        if trades < 30: # Force some volume
            return -999.0
            
        winrate = (wins/trades) if trades > 0 else 0
        return total_pnl

def main():
    df, close_prices, high_prices, low_prices, all_probs, s_len = precompute_signals()
    
    print("Running Fast Optuna strictly on TP and SL for SHORT trial_270 ...")
    study = optuna.create_study(direction='maximize')
    study.optimize(FastShortObjective(close_prices, high_prices, low_prices, all_probs, s_len), n_trials=500)
    
    best = study.best_trial
    print(f"\nBest Parameters for SHORT Model:")
    print(f"Total Evaluated Short PnL Score: {best.value}%")
    for key, val in best.params.items():
        print(f"  {key}: {val}")

if __name__ == '__main__':
    main()
