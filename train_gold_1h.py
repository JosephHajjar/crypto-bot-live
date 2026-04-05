import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import optuna
import json
import urllib.request
import time
from data.fetch_data import fetch_klines
from data.feature_engineer import precompute_static_features, dynamic_features_and_labels, get_feature_cols
from ml.dataset import TimeSeriesDataset
from ml.model import AttentionLSTMModel

# ================================================================
# CONFIGURATION
# ================================================================
SYMBOL = 'PAXG/USDT'
TIMEFRAME = '1h'
DAYS = 1000
DATA_DIR = 'data_storage'
MODEL_DIR = 'models_gold_1h'
STATE_FILE = 'optuna_gold_1h_state.json'
LOG_FILE = 'optuna_gold_1h.log'

NUM_FOLDS = 3
FEE_PCT = 0.001

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'a', buffering=1, encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ================================================================
# SHARPE RATIO BACKTESTER
# ================================================================
def evaluate_strategy(df_slice, model, seq_length, device, label_horizon, label_threshold, fee_pct=FEE_PCT):
    """
    Backtests the 3-class model evaluating both Long (+1) and Short (-1) trades using ATR barriers.
    Returns Sharpe/Calmar and stats.
    """
    feature_cols = get_feature_cols()
    X = df_slice[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X)
    
    close = df_slice['close'].values
    high = df_slice['high'].values
    low = df_slice['low'].values
    atr = df_slice['ATR'].values
    
    if len(X) < seq_length + label_horizon + 10:
        return {'sharpe': -10.0, 'calmar': -10.0, 'trades': 0}
        
    all_seqs = np.array([X[i:i+seq_length] for i in range(len(X) - seq_length)])
    seq_tensor = torch.tensor(all_seqs, dtype=torch.float32).to(device)
    
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seq_tensor), 2048):
            batch = seq_tensor[i:i+2048]
            out = model(batch)
            preds = torch.argmax(out, dim=1).cpu().tolist()
            predictions.extend(preds)
            
    capital = 10000.0
    max_capital = capital
    max_drawdown = 0.0
    
    trade_returns = []
    i = 0
    
    while i < len(predictions):
        sig_idx = i + seq_length
        if sig_idx + label_horizon >= len(close): break
            
        pred = predictions[i]
        
        if pred == 1: # LONG
            entry_price = close[sig_idx]
            barrier = atr[sig_idx] * label_threshold
            tp_price = entry_price + barrier
            sl_price = entry_price - barrier
            exit_price = close[sig_idx + label_horizon] # default time exit
            
            for j in range(1, label_horizon + 1):
                idx = sig_idx + j
                if idx >= len(close): break
                if low[idx] <= sl_price:
                    exit_price = sl_price
                    break
                if high[idx] >= tp_price:
                    exit_price = tp_price
                    break
            
            ret = ((exit_price - entry_price) / entry_price) - (fee_pct * 2)
            trade_returns.append(ret)
            capital *= (1 + ret)
            i += label_horizon
            
        elif pred == 2: # SHORT
            entry_price = close[sig_idx]
            barrier = atr[sig_idx] * label_threshold
            tp_price = entry_price - barrier 
            sl_price = entry_price + barrier
            exit_price = close[sig_idx + label_horizon]
            
            for j in range(1, label_horizon + 1):
                idx = sig_idx + j
                if idx >= len(close): break
                if high[idx] >= sl_price:
                    exit_price = sl_price
                    break
                if low[idx] <= tp_price:
                    exit_price = tp_price
                    break
                    
            ret = ((entry_price - exit_price) / entry_price) - (fee_pct * 2)
            trade_returns.append(ret)
            capital *= (1 + ret)
            i += label_horizon
            
        else:
            i += 1
            
        if capital > max_capital: max_capital = capital
        dd = (capital - max_capital) / max_capital
        if dd < max_drawdown: max_drawdown = dd

    num_trades = len(trade_returns)
    if num_trades < 10:
        return {'sharpe': -10.0, 'calmar': -10.0, 'trades': num_trades, 'roi': 0, 'drawdown': 0}
        
    avg_ret = np.mean(trade_returns)
    std_ret = np.std(trade_returns)
    
    # Approx 24 hours * 252 trading days = ~6048 hours/yr
    annual_factor = np.sqrt(6048 / max(1, label_horizon)) 
    sharpe = (avg_ret / std_ret) * annual_factor if std_ret > 0 else 0
    
    roi = (capital - 10000.0) / 10000.0
    calmar = roi / abs(max_drawdown) if max_drawdown < 0 else 0
    
    return {
        'sharpe': sharpe,
        'calmar': calmar,
        'trades': num_trades,
        'roi': roi * 100,
        'drawdown': max_drawdown * 100
    }

def walk_forward_evaluate(df, model, seq_length, device, label_horizon, label_threshold):
    n = len(df)
    fold_size = n // 4
    
    metrics = []
    
    for fold in range(3):
        start = fold_size * (fold + 1)
        end = start + fold_size
        df_fold = df.iloc[start:end]
        res = evaluate_strategy(df_fold, model, seq_length, device, label_horizon, label_threshold)
        metrics.append(res)
        
    if not metrics or metrics[0]['trades'] < 5:
        return -10.0
        
    avg_sharpe = np.mean([m['sharpe'] for m in metrics])
    avg_calmar = np.mean([m['calmar'] for m in metrics])
    min_trades = np.min([m['trades'] for m in metrics])
    
    if min_trades < 5: return -5.0
    
    # We return Sharpe directly for maximization
    return avg_sharpe

# ================================================================
# OPTUNA OBJECTIVE
# ================================================================
def create_objective(static_csv_path, train_end, val_end, device):
    print("Loading static precomputed features into RAM...")
    df_static = pd.read_csv(static_csv_path, index_col='timestamp', parse_dates=True)
    
    # Pre-scale purely static features from training split
    train_slice = df_static.iloc[:train_end]
    feature_cols_all = get_feature_cols()
    static_cols = [c for c in feature_cols_all if c not in ['ATR_Norm', 'RSI']]
    
    means = train_slice[static_cols].mean()
    stds = train_slice[static_cols].std().replace(0, 1)
    df_static[static_cols] = (df_static[static_cols] - means) / stds

    def objective(trial):
        # 1. Feature Hyperparameters
        atr_period = trial.suggest_int('atr_period', 10, 21)
        rsi_period = trial.suggest_int('rsi_period', 10, 21)
        label_horizon = trial.suggest_int('label_horizon', 4, 24)
        label_threshold = trial.suggest_float('label_threshold', 1.0, 2.5)
        
        # 2. NN Architecture
        n_layers = trial.suggest_int('n_layers', 2, 5)
        hidden_size = trial.suggest_int('hidden_size', 64, 512)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'leaky_relu'])
        
        # 3. Training
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
        
        print(f"\n[Trial #{trial.number}] Arch: {n_layers}L {hidden_size}H, {activation} | "
              f"Feat: ATR{atr_period} RSI{rsi_period} Hor{label_horizon} Thresh{label_threshold:.2f}")

        # Compute dynamic features instantly
        df_dyn = dynamic_features_and_labels(df_static, atr_period, rsi_period, label_horizon, label_threshold)
        
        # Scale the new dynamic columns using training split
        dyn_cols = ['ATR_Norm', 'RSI']
        dyn_means = df_dyn.iloc[:train_end][dyn_cols].mean()
        dyn_stds = df_dyn.iloc[:train_end][dyn_cols].std().replace(0, 1)
        df_dyn[dyn_cols] = (df_dyn[dyn_cols] - dyn_means) / dyn_stds
        
        # Build dataset
        seq_len = 64
        dataset = TimeSeriesDataset('dummy', seq_length=seq_len)
        dataset.data = df_dyn[feature_cols_all].values.astype(np.float32)
        dataset.data = np.nan_to_num(dataset.data)
        dataset.y = df_dyn['Target'].values.astype(np.int64)
        dataset.feature_cols = feature_cols_all
        
        train_indices = list(range(min(train_end - seq_len, len(dataset))))
        train_sampler = torch.utils.data.SequentialSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True
        )

        model = AttentionLSTMModel(
            input_dim=len(feature_cols_all), hidden_dim=hidden_size,
            num_layers=n_layers, output_dim=3, dropout=dropout, activation_fn=activation
        ).to(device)

        # Handle class imbalance heavily for 3 classes
        class_counts = df_dyn.iloc[:train_end]['Target'].value_counts().to_dict()
        total = sum(class_counts.values())
        weights = torch.tensor([total / max(1, class_counts.get(i, 1)) for i in range(3)], dtype=torch.float32)
        weights = weights / weights.sum() # Normalize
        
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        opt_class = optim.Adam if optimizer_name == 'adam' else optim.AdamW
        optimizer = opt_class(model.parameters(), lr=lr)

        model.train()
        epochs = 15
        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

        # Evaluate Sharpe
        df_val = df_dyn.iloc[train_end:val_end]
        val_sharpe = walk_forward_evaluate(df_val, model, seq_len, device, label_horizon, label_threshold)
        
        print(f"  --> Val Sharpe: {val_sharpe:.3f}")
        
        if val_sharpe > 1.0:
            os.makedirs(MODEL_DIR, exist_ok=True)
            path = f"{MODEL_DIR}/trial_{trial.number}_sharpe_{val_sharpe:.2f}.pth"
            torch.save(model.state_dict(), path)
            print(f"  [!] Edge Found! Model saved to {path}")
            
        return val_sharpe

    return objective

def main():
    sys.stdout = Logger(LOG_FILE)
    print("="*60)
    print("GOLD MACRO OPTIMIZATION: 1H TIMEFRAME")
    print("="*60)
    
    csv_path = fetch_klines(SYMBOL, TIMEFRAME, DAYS, DATA_DIR, fetch_macro=True)
    static_csv_path = precompute_static_features(csv_path)
    
    df = pd.read_csv(static_csv_path)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Data Base: {n} hours | Train: {train_end} | Val: {val_end-train_end}")
    
    study = optuna.create_study(direction='maximize', study_name='gold_1h_sharpe', storage='sqlite:///gold_1h.db', load_if_exists=True)
    objective = create_objective(static_csv_path, train_end, val_end, device)
    
    study.optimize(objective, n_trials=50, show_progress_bar=False)

if __name__ == "__main__":
    main()
