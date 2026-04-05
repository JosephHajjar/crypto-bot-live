import os
import json
import torch
import numpy as np
import pandas as pd
import optuna

from ml.model import AttentionLSTMModel
from data.feature_engineer import get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def precompute_signals():
    print("Precomputing signals for trial_255 TSL...")
    df = pd.read_csv('data_storage/BTC_USDT_15m_processed.csv')
    df = df.iloc[-8000:].copy() # Test on 8000 bars
    
    with open('models/trial_255_config.json', 'r') as f:
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
    model.load_state_dict(torch.load('models/trial_255.pth', map_location=device, weights_only=True))
    model.eval()
    
    batch_size = 256
    all_probs = []
    
    with torch.no_grad():
        for start in range(0, len(features_np) - s_len + 1, batch_size):
            end = min(start + batch_size, len(features_np) - s_len + 1)
            batch = [features_np[i : i + s_len] for i in range(start, end)]
            tensor_batch = torch.tensor(np.array(batch)).to(device)
            out = model(tensor_batch)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy().tolist()
            all_probs.extend(probs)
            
    return df, close_prices, high_prices, low_prices, all_probs, s_len

class TrailingObjective:
    def __init__(self, close_prices, high_prices, low_prices, all_probs, s_len):
        self.close_prices = close_prices
        self.high_prices = high_prices
        self.low_prices = low_prices
        self.all_probs = np.array(all_probs)
        self.s_len = s_len
        
    def __call__(self, trial):
        init_sl_pct = trial.suggest_float('initial_sl', 0.005, 0.05, step=0.005)
        activation_pct = trial.suggest_float('activation_pct', 0.005, 0.10, step=0.005)
        trailing_pct = trial.suggest_float('trailing_pct', 0.005, 0.05, step=0.005)
        max_hold_bars = trial.suggest_int('max_hold_bars', 4, 96, step=4)
        
        position = 'flat'
        entry_price = 0
        bars_held = 0
        trades = 0
        total_pnl = 0
        wins = 0
        
        highest_price = 0
        tsl_active = False
        sl_price = 0
        
        for idx in range(len(self.all_probs)):
            prob_bull = self.all_probs[idx]
            bar_idx = self.s_len - 1 + idx
            
            if bar_idx < len(self.close_prices):
                close_price = self.close_prices[bar_idx]
                high_price = self.high_prices[bar_idx]
                low_price = self.low_prices[bar_idx]
                
                if position == 'flat':
                    if prob_bull >= 0.50:
                        position = 'long'
                        entry_price = close_price
                        sl_price = entry_price * (1 - init_sl_pct)
                        highest_price = high_price
                        tsl_active = False
                        bars_held = 0
                else:
                    bars_held += 1
                    closed = False
                    pnl = 0
                    
                    # Track Highest Watermark
                    if high_price > highest_price:
                        highest_price = high_price
                        
                    # Check if Trailing Activates
                    activation_target = entry_price * (1 + activation_pct)
                    if highest_price >= activation_target:
                        tsl_active = True
                        
                    # Adjust SL if Trailing is Active
                    if tsl_active:
                        proposed_sl = highest_price * (1 - trailing_pct)
                        # Ratchet logic: TSL can only ever move UP
                        if proposed_sl > sl_price:
                            sl_price = proposed_sl
                            
                    # Check for knock-out
                    if low_price <= sl_price:
                        # Assuming worst case slippage, closed at exact sl_price threshold
                        pnl = (sl_price - entry_price) / entry_price * 100
                        closed = True
                    elif bars_held >= max_hold_bars:
                        pnl = (close_price - entry_price) / entry_price * 100
                        closed = True
                        
                    if closed:
                        total_pnl += pnl
                        trades += 1
                        if pnl > 0: wins += 1
                        position = 'flat'
                        
        if trades < 30: 
            return -999.0
            
        # Give higher weight to ROI to maximize absolute PnL extraction
        winrate = (wins/trades) if trades > 0 else 0
        return total_pnl

def main():
    df, close, high, low, probs, s_len = precompute_signals()
    
    print("Running Fast Optuna with Trailing Stop-Loss for trial_255...")
    study = optuna.create_study(direction='maximize')
    study.optimize(TrailingObjective(close, high, low, probs, s_len), n_trials=500)
    
    best = study.best_trial
    print(f"\nBest Parameters for Trial 255 (Trailing Stop Loss):")
    print(f"Total Combined PnL Evaluation Score: {best.value}%")
    for key, val in best.params.items():
        print(f"  {key}: {val}")

if __name__ == '__main__':
    main()
