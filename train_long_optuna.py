import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import optuna
import shutil
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

# Import ML model
from ml.model import AttentionLSTMModel
from optimize_short_threshold import fetch_recent_15m
from data.feature_engineer import compute_live_features, get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def backtest_long_vectorized(signals, close, high, low, seq_len, TP, SL, MAX_BARS):
    capital = 10000.0
    initial = capital
    trades = []
    i = 0
    FEE = 0.001
    
    while i < len(signals):
        idx = i + seq_len
        if idx + MAX_BARS >= len(close): break
        
        prob = signals[i]
        if prob > 0.50:
            entry = close[idx]
            entry_cap = capital
            tp_price = entry * (1 + TP)
            sl_price = entry * (1 - SL)
            
            exit_p = None
            for j in range(1, MAX_BARS + 1):
                jdx = idx + j
                if jdx >= len(close): break
                if low[jdx] <= sl_price:
                    exit_p = sl_price; break
                if high[jdx] >= tp_price:
                    exit_p = tp_price; break
                    
            if exit_p is None:
                exit_p = close[idx + MAX_BARS]
                
            ret = (exit_p - entry) / entry - FEE * 2
            capital = entry_cap * (1 + ret)
            trades.append({'win': ret > 0, 'ret': ret})
            i += MAX_BARS
        else:
            i += 1
            
    roi = (capital - initial) / initial * 100
    if len(trades) > 1:
        rets = [t['ret'] for t in trades]
        sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(len(trades))
    else:
        sharpe = 0
        
    wins = sum(1 for t in trades if t['win'])
    wl = f"{wins}/{len(trades)-wins}"
    return roi, sharpe, len(trades), wl

def objective(trial):
    # Hyperparameters to tune
    seq_len = trial.suggest_int('seq_len', 32, 128, step=16)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    num_heads = trial.suggest_categorical('num_heads', [4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    # Validation constraints / Risk management params
    take_profit = trial.suggest_float('take_profit', 0.01, 0.15, step=0.005)
    stop_loss = trial.suggest_float('stop_loss', 0.005, 0.05, step=0.005)
    max_hold_bars = trial.suggest_int('max_hold_bars', 4, 24, step=4)
    
    # Global datasets setup
    global X_train, y_train, X_val, y_val, close_val, high_val, low_val, input_dim
    
    # Re-slice arrays based on seq_len
    # Training Data
    X_tr_seq, y_tr_seq = [], []
    for i in range(len(X_train) - seq_len):
        X_tr_seq.append(X_train[i:i+seq_len])
        y_tr_seq.append(y_train[i+seq_len])
        
    X_tr_tensor = torch.tensor(np.array(X_tr_seq), dtype=torch.float32).to(device)
    y_tr_tensor = torch.tensor(np.array(y_tr_seq), dtype=torch.long).to(device)
    
    train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialization
    model = AttentionLSTMModel(
        input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
        output_dim=2, dropout=dropout, num_heads=num_heads
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Train limited epochs for optuna evaluation
    epochs = 4
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    # Validation Evaluation
    model.eval()
    X_val_seq = np.array([X_val[i:i+seq_len] for i in range(len(X_val) - seq_len)])
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
    
    signals = []
    with torch.no_grad():
        for i in range(0, len(X_val_tensor), 2048):
            batch = X_val_tensor[i:i+2048]
            out = model(batch)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().tolist()
            signals.extend(probs)
            
    # Backtest using dynamic SL/TP
    roi, sharpe, trades, wl = backtest_long_vectorized(
        signals, close_val, high_val, low_val, seq_len, 
        take_profit, stop_loss, max_hold_bars
    )
    
    # Save the output weights directly for analysis
    trial_name = f"trial_long_{trial.number}"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    if roi > 0 and trades > 10:
        cfg = {
            'trial': trial.number,
            'seq_len': seq_len, 'hidden_dim': hidden_dim, 'num_layers': num_layers,
            'num_heads': num_heads, 'dropout': dropout, 'input_dim': input_dim,
            'take_profit': take_profit, 'stop_loss': stop_loss, 'max_hold_bars': max_hold_bars,
            'val_roi': roi, 'val_sharpe': sharpe, 'trades': trades, 'wl': wl
        }
        with open(os.path.join(model_dir, f"{trial_name}_config.json"), 'w') as f:
            json.dump(cfg, f, indent=2)
        torch.save(model.state_dict(), os.path.join(model_dir, f"{trial_name}.pth"))
        
    # Standardize objective score. High ROI with constraints.
    if trades < 5: return -999.0
    return roi + (sharpe * 10)

def main():
    print("Pre-fetching dataset for Optuna Loop...")
    df_raw = fetch_recent_15m(200) # Past 200 days for train & validation
    df_feat = compute_live_features(df_raw, 'data_storage/BTC_USDT_15m_scaler.json')
    feature_cols = get_feature_cols()
    
    # We define a bullish target to train network. Example: Return > 0.005
    future_rets = df_feat['close'].shift(-4) / df_feat['close'] - 1.0
    df_feat['target'] = (future_rets > 0.005).astype(int)
    
    df_feat = df_feat.dropna().reset_index(drop=True)
    
    # Split
    split_idx = int(len(df_feat) * 0.70)
    train_df = df_feat.iloc[:split_idx]
    val_df = df_feat.iloc[split_idx:]
    
    global X_train, y_train, X_val, y_val, close_val, high_val, low_val, input_dim
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['target'].values
    
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['target'].values
    
    close_val = val_df['close'].values
    high_val = val_df['high'].values
    low_val = val_df['low'].values
    input_dim = X_train.shape[1]
    
    print(f"Target distribution train: {y_train.sum()}/{len(y_train)}")
    
    study = optuna.create_study(direction='maximize', study_name="Long_Optimization")
    study.optimize(objective, n_trials=50)
    
    best = study.best_trial
    print(f"\nBest Trial: {best.number}")
    print(f"  ROI Score: {best.value}")
    for key, val in best.params.items():
        print(f"    {key}: {val}")

if __name__ == '__main__':
    main()
