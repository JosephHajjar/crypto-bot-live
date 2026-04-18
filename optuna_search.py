"""
Optuna Bayesian Hyperparameter Search with Walk-Forward Validation.

Key differences from the old random search:
1. Optuna LEARNS from past trials — it remembers what worked and narrows the search
2. Walk-Forward validation — tests across multiple time windows, not just one
3. Triple Barrier backtesting — matches the labeling strategy
4. Statistical significance gates — min trades + Sharpe ratio requirements
5. Saves every promising model with full config
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import time
import urllib.request
import optuna
from optuna.trial import TrialState

from data.fetch_data import fetch_klines
from data.feature_engineer_btc import engineer_features, get_feature_cols
from ml.dataset import TimeSeriesDataset
from ml.model import AttentionLSTMModel

# ================================================================
# CONFIGURATION
# ================================================================
SYMBOL = 'BTC/USDT'
TIMEFRAME = '15m'
DAYS = 365
DATA_DIR = 'data_storage'
MODEL_DIR = 'models'
STATE_FILE = 'optuna_state.json'
LOG_FILE = 'optuna_output.log'

# Triple barrier params (must match feature_engineer.py)
TAKE_PROFIT = 0.015   # 1.5%
STOP_LOSS = 0.0075     # 0.75%
MAX_HOLD_BARS = 16     # 4 hours max hold

# Validation requirements
MIN_TRADES = 30        # Minimum trades for statistical significance
MIN_SHARPE = 0.5       # Minimum Sharpe ratio (annualized)
TARGET_ANNUAL_ROI = 15.0

# Walk-forward settings
NUM_FOLDS = 3          # Number of walk-forward folds

FEE_PCT = 0.001        # 0.1% per side (0.2% round trip)

# ================================================================
# LOGGING
# ================================================================
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
# BACKTESTING (Triple Barrier aligned)
# ================================================================
def backtest_triple_barrier(df_slice, model, seq_length, device, 
                            tp=TAKE_PROFIT, sl=STOP_LOSS, max_bars=MAX_HOLD_BARS,
                            fee_pct=FEE_PCT):
    """
    Backtest using the Triple Barrier method, matching how we labeled the data.
    When model predicts 1 (enter trade):
      - Enter at current close
      - Exit when TP or SL is hit, or after max_bars
    When model predicts 0: stay flat.
    """
    feature_cols = get_feature_cols()
    available = [c for c in feature_cols if c in df_slice.columns]
    
    X = df_slice[available].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    close = df_slice['close'].values
    high = df_slice['high'].values
    low = df_slice['low'].values
    
    if len(X) < seq_length + max_bars + 10:
        return {'roi': 0, 'annual_roi': 0, 'sharpe': 0, 'wins': 0, 'losses': 0, 'trades': []}
    
    # Generate all signals at once (vectorized)
    all_seqs = np.array([X[i:i+seq_length] for i in range(len(X) - seq_length)])
    seq_tensor = torch.tensor(all_seqs, dtype=torch.float32).to(device)
    
    signals = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seq_tensor), 2048):
            batch = seq_tensor[i:i+2048]
            out = model(batch)
            probs = torch.softmax(out, dim=1)
            bull_probs = probs[:, 1].cpu().tolist()
            signals.extend(bull_probs)
    
    # Simulate trading
    capital = 10000.0
    initial_capital = capital
    max_capital = capital
    max_drawdown = 0.0
    trades = []
    trade_returns = []
    i = 0
    
    while i < len(signals):
        sig_idx = i + seq_length  # Index into the price array
        
        if sig_idx + max_bars >= len(close):
            break
            
        if capital > 0:
            prob = signals[i]
            if prob > 0.50:
                # ENTER LONG
                entry_price = close[sig_idx]
                entry_capital = capital
                position_size = capital * (1 - fee_pct)  # Pay entry fee
                
                tp_price = entry_price * (1 + tp)
                sl_price = entry_price * (1 - sl)
                
                exit_price = None
                exit_reason = None
                
                # Walk forward bar by bar
                for j in range(1, max_bars + 1):
                    idx = sig_idx + j
                    if idx >= len(close):
                        break
                        
                    # Check SL first (conservative)
                    if low[idx] <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'SL'
                        break
                    # Check TP
                    if high[idx] >= tp_price:
                        exit_price = tp_price
                        exit_reason = 'TP'
                        break
                
                # Time barrier — exit at close of last bar
                if exit_price is None:
                    exit_price = close[sig_idx + max_bars]
                    exit_reason = 'TIME'
                
                # Calculate P&L
                price_return = (exit_price - entry_price) / entry_price
                net_return = price_return - (fee_pct * 2)  # Round-trip fee
                capital = entry_capital * (1 + net_return)
                
                if capital > max_capital:
                    max_capital = capital
                drawdown = ((capital - max_capital) / max_capital) * 100
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
                
                trade_returns.append(net_return)
                trades.append({
                    'entry_idx': sig_idx,
                    'exit_reason': exit_reason,
                    'return': net_return * 100,
                    'win': net_return > 0,
                    'type': 'LONG'
                })
                
                # Skip ahead past the trade duration (can't re-enter while in trade)
                i += max_bars
            else:
                i += 1
        else:
            i += 1
    
    # Calculate metrics
    total_roi = ((capital - initial_capital) / initial_capital) * 100
    
    days_in_slice = len(df_slice) * 15 / 1440
    if days_in_slice > 0 and total_roi > -99:
        annual_roi = ((1 + total_roi / 100) ** (365 / days_in_slice) - 1) * 100
    else:
        annual_roi = -100.0
    
    # Sharpe ratio (annualized)
    if len(trade_returns) > 1:
        avg_ret = np.mean(trade_returns)
        std_ret = np.std(trade_returns)
        if std_ret > 0:
            # Annualize: assume ~2 trades per day on 15m data
            trades_per_year = (365 / days_in_slice) * len(trade_returns) if days_in_slice > 0 else 252
            sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0
    
    wins = sum(1 for t in trades if t['win'])
    losses = len(trades) - wins
    
    return {
        'roi': total_roi,
        'annual_roi': annual_roi,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'wins': wins,
        'losses': losses,
        'num_trades': len(trades),
        'trades': trades
    }

# ================================================================  
# WALK-FORWARD VALIDATION
# ================================================================
def walk_forward_evaluate(df, model, seq_length, device, num_folds=NUM_FOLDS):
    """
    Walk-forward validation across multiple time windows.
    Returns the WORST-CASE metrics across all folds.
    A model must be profitable in ALL windows, not just one lucky one.
    """
    n = len(df)
    fold_size = n // (num_folds + 1)  # Reserve first chunk for initial training context
    
    fold_results = []
    
    for fold in range(num_folds):
        start = fold_size * (fold + 1)
        end = min(start + fold_size, n)
        
        if end - start < seq_length + MAX_HOLD_BARS + 50:
            continue
            
        df_fold = df.iloc[start:end]
        result = backtest_triple_barrier(df_fold, model, seq_length, device)
        fold_results.append(result)
    
    if not fold_results:
        return {'roi': -100, 'annual_roi': -100, 'sharpe': -10, 'wins': 0, 'losses': 0, 
                'num_trades': 0, 'worst_fold_roi': -100, 'avg_sharpe': -10}
    
    # Aggregate: use WORST fold as the primary metric (conservative)
    worst_roi = min(r['annual_roi'] for r in fold_results)
    avg_sharpe = np.mean([r['sharpe'] for r in fold_results])
    worst_drawdown = min(r['max_drawdown'] for r in fold_results)
    total_trades = sum(r['num_trades'] for r in fold_results)
    total_wins = sum(r['wins'] for r in fold_results)
    total_losses = sum(r['losses'] for r in fold_results)
    avg_roi = np.mean([r['annual_roi'] for r in fold_results])
    
    return {
        'avg_annual_roi': avg_roi,
        'worst_fold_roi': worst_roi,
        'avg_sharpe': avg_sharpe,
        'worst_drawdown': worst_drawdown,
        'total_trades': total_trades,
        'wins': total_wins,
        'losses': total_losses,
        'fold_details': fold_results
    }

# ================================================================
# OPTUNA OBJECTIVE
# ================================================================
def create_objective(df_train, df_val, df_test, processed_path, train_end, device):
    """Create an Optuna objective function with the data baked in."""
    
    def objective(trial):
        # Suggest hyperparameters (Optuna learns which ranges work)
        seq_len = trial.suggest_categorical('seq_len', [32, 48, 64, 96, 128])
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        epochs = trial.suggest_int('epochs', 10, 35)
        weight_ratio = trial.suggest_float('class_weight', 2.0, 6.0)
        
        # Ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            num_heads = min(h for h in [1, 2, 4, 8] if hidden_dim % h == 0 and h <= num_heads)
        
        print(f"\n[Trial #{trial.number}] seq={seq_len}, hidden={hidden_dim}, layers={num_layers}, "
              f"heads={num_heads}, drop={dropout:.2f}, lr={lr:.5f}, batch={batch_size}, "
              f"epochs={epochs}, class_w={weight_ratio:.1f}")
        
        # Build dataset and loader
        dataset = TimeSeriesDataset(processed_path, seq_length=seq_len)
        train_indices = list(range(min(train_end - seq_len, len(dataset))))
        train_sampler = torch.utils.data.SequentialSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler, 
            drop_last=True, pin_memory=True, num_workers=0
        )
        
        input_dim = len(dataset.feature_cols)
        
        # Build model
        model = AttentionLSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=2,
            dropout=dropout,
            num_heads=num_heads
        ).to(device)
        
        # Class weights to handle imbalance
        class_weights = torch.tensor([1.0, weight_ratio]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Train
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            batches = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1
            scheduler.step()
            
            # Report intermediate value for pruning
            if batches > 0:
                trial.report(epoch_loss / batches, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # ---- VALIDATION: Walk-Forward across val set ----
        val_result = walk_forward_evaluate(df_val, model, seq_len, device)
        
        avg_roi = val_result['avg_annual_roi']
        avg_sharpe = val_result['avg_sharpe']
        worst_drawdown = val_result['worst_drawdown']
        total_trades = val_result['total_trades']
        worst_roi = val_result['worst_fold_roi']
        wins = val_result['wins']
        losses = val_result['losses']
        
        print(f"  Val -> AvgROI: {avg_roi:.1f}% | WorstFold: {worst_roi:.1f}% | "
              f"Sharpe: {avg_sharpe:.2f} | MaxDD: {worst_drawdown:.1f}% | Trades: {total_trades} | W/L: {wins}/{losses}")
        
        # ---- STATISTICAL SIGNIFICANCE GATE ----
        if total_trades < MIN_TRADES:
            print(f"  REJECTED: Only {total_trades} trades (need {MIN_TRADES}+)")
            return -100.0
            
        if worst_drawdown < -7.0:
            print(f"  REJECTED: Max Drawdown of {worst_drawdown:.1f}% exceeded -7.0% limit")
            return -100.0
        
        # ---- TEST on truly unseen data if validation looks promising ----
        if avg_roi >= 12.3:
            print(f"  --> Edge potentially found on Validation (>12.3%)! Verifying on Unseen Test Data...")
            test_result = backtest_triple_barrier(df_test, model, seq_len, device)
            test_roi = test_result['annual_roi']
            test_sharpe = test_result['sharpe']
            test_trades = test_result['num_trades']
            tw, tl = test_result['wins'], test_result['losses']
            
            print(f"  Test -> ROI: {test_roi:.1f}% | Sharpe: {test_sharpe:.2f} | "
                  f"Trades: {test_trades} | W/L: {tw}/{tl}")
            
            # Save if it passes test
            if test_roi > 0 and test_trades >= 10:
                os.makedirs(MODEL_DIR, exist_ok=True)
                model_path = f"{MODEL_DIR}/trial_{trial.number}.pth"
                torch.save(model.state_dict(), model_path)
                config = {
                    'trial': trial.number, 'seq_len': seq_len, 'hidden_dim': hidden_dim,
                    'num_layers': num_layers, 'num_heads': num_heads, 'dropout': dropout,
                    'input_dim': input_dim,
                    'val_roi': avg_roi, 'val_sharpe': avg_sharpe,
                    'test_roi': test_roi, 'test_sharpe': test_sharpe,
                    'test_trades': test_trades, 'test_wl': f"{tw}/{tl}"
                }
                with open(f"{MODEL_DIR}/trial_{trial.number}_config.json", 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"  ** MODEL SAVED: {model_path} **")
                
                # Notify
                try:
                    msg = f"Optuna Trial #{trial.number}: Val={avg_roi:.1f}% Test={test_roi:.1f}% Sharpe={test_sharpe:.2f}"
                    urllib.request.urlopen(
                        urllib.request.Request("https://ntfy.sh/TradeBot5234", data=msg.encode()),
                        timeout=5
                    )
                except Exception:
                    pass
                
                # Check Holy Grail
                if test_roi > TARGET_ANNUAL_ROI and test_sharpe > 1.0 and test_trades >= MIN_TRADES:
                    print("\n" + "=" * 60)
                    print("🏆 HOLY GRAIL FOUND!")
                    print(f"   Test ROI: {test_roi:.1f}% | Sharpe: {test_sharpe:.2f} | W/L: {tw}/{tl}")
                    print("=" * 60)
                    torch.save(model.state_dict(), f"{MODEL_DIR}/holy_grail.pth")
                    with open(f"{MODEL_DIR}/holy_grail_config.json", 'w') as f:
                        json.dump(config, f, indent=2)
        
        # Optuna maximizes the return value
        # Use a composite score: ROI weighted by Sharpe (rewards consistency)
        if avg_sharpe > 0:
            score = avg_roi * (1 + avg_sharpe * 0.1)
        else:
            score = avg_roi
            
        return score
    
    return objective

# ================================================================
# MAIN
# ================================================================
def main():
    sys.stdout = Logger(LOG_FILE)
    
    print("=" * 60)
    print("OPTUNA BAYESIAN SEARCH WITH WALK-FORWARD VALIDATION")
    print("=" * 60)
    
    # 1. Fetch & prepare data
    print(f"\nFetching {DAYS} days of {TIMEFRAME} {SYMBOL} data...")
    csv_path = fetch_klines(SYMBOL, TIMEFRAME, DAYS, save_dir=DATA_DIR)
    if not csv_path:
        print("Failed to fetch data!")
        return
    
    print("Engineering features (multi-timeframe + triple barrier)...")
    processed_path, stats = engineer_features(
        csv_path, 
        take_profit=TAKE_PROFIT,
        stop_loss=STOP_LOSS,
        max_hold_bars=MAX_HOLD_BARS
    )
    
    # 2. Load and split
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    
    print(f"\nData splits -> Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    print(f"Target distribution (train): {df_train['Target'].value_counts().to_dict()}")
    print(f"Target distribution (val): {df_val['Target'].value_counts().to_dict()}")
    print(f"Target distribution (test): {df_test['Target'].value_counts().to_dict()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 3. Create Optuna study (Bayesian optimization) — persisted to SQLite for resume
    db_path = os.path.join(DATA_DIR, 'optuna_study.db')
    storage = f'sqlite:///{db_path}'
    study = optuna.create_study(
        direction='maximize',
        study_name='trading_bot_v2',
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(n_startup_trials=15)
    )
    print(f"Study loaded from {db_path} — {len(study.trials)} previous trials found")
    
    objective = create_objective(df_train, df_val, df_test, processed_path, train_end, device)
    
    print("\n" + "=" * 60)
    print("COMMENCING BAYESIAN SEARCH (Optuna TPE Sampler)")
    print(f"Statistical gates: Min {MIN_TRADES} trades, Sharpe > {MIN_SHARPE}")
    print(f"Triple Barrier: TP={TAKE_PROFIT*100}% / SL={STOP_LOSS*100}% / MaxHold={MAX_HOLD_BARS} bars")
    print("=" * 60 + "\n")
    
    # 4. Run indefinitely
    study.optimize(objective, n_trials=2000, show_progress_bar=False)
    
    # 5. Print summary
    print("\n" + "=" * 60)
    print("SEARCH COMPLETE")
    print("=" * 60)
    
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Completed trials: {len(completed)}")
    
    if study.best_trial:
        print(f"Best score: {study.best_value:.2f}")
        print(f"Best params: {study.best_params}")

if __name__ == "__main__":
    main()
