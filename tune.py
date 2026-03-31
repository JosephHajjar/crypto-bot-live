import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

from ml.dataset import TimeSeriesDataset
from ml.model import LSTMTradingModel, save_model

def calculate_roi(df_slice, model, seq_length, device, fee_pct=0.001):
    """
    Evaluates trading model over a specific DataFrame slice.
    """
    # Features need to match dataset exactly
    feature_cols = [
        'Returns', 'SMA_10', 'SMA_50', 'RSI_14', 
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
        'ATR_14', 'Vol_Ratio'
    ]
    
    X_val = df_slice[feature_cols].values
    close_prices = df_slice['close'].values
    
    capital = 10000.0
    initial_capital = capital
    position = 0 
    entry_capital = 0
    trades_won = 0
    trades_lost = 0
    
    length = len(X_val) - 1 - seq_length
    if length <= 0: return 0.0, 0.0, 0, 0
        
    all_seqs = [X_val[i-seq_length : i] for i in range(seq_length, len(X_val) - 1)]
    seq_tensor = torch.tensor(np.array(all_seqs), dtype=torch.float32).to(device)
    
    signals = []
    model.eval()
    with torch.no_grad():
        batch_size = 2048
        for i in range(0, len(seq_tensor), batch_size):
            batch = seq_tensor[i : i+batch_size]
            out = model(batch)
            _, pred = torch.max(out, 1)
            signals.extend(pred.cpu().tolist())
            
    for idx, i in enumerate(range(seq_length, len(X_val) - 1)):
        signal = signals[idx]
        current_return = (close_prices[i+1] - close_prices[i]) / close_prices[i]
        
        if signal == 1 and position == 0:
            # Buy
            position = capital * (1 - fee_pct)
            capital = 0
            entry_capital = position / (1 - fee_pct)
        elif signal == 0 and position > 0:
            # Sell
            capital = position * (1 - fee_pct)
            if capital > entry_capital:
                 trades_won += 1
            else:
                 trades_lost += 1
            position = 0
            
        if position > 0:
            position = position * (1 + current_return)
                
    if position > 0:
         capital = position * (1 - fee_pct)
         
    total_profit = capital - initial_capital
    roi = (total_profit / initial_capital) * 100
    
    # Calculate equivalent Annualized Return for benchmarking
    # Days in slice = length * 5 minutes / (60 * 24)
    days_in_slice = len(df_slice) * 5 / 1440
    if days_in_slice > 0:
        annual_roi = ((1 + roi/100) ** (365 / days_in_slice) - 1) * 100
    else:
        annual_roi = 0.0
        
    return roi, annual_roi, trades_won, trades_lost

def optimize_bot():
    print("=== Commencing Walk-Forward Optimization Loop ===")
    data_path = 'data_storage/BTC_USDT_5m_processed.csv'
    
    if not os.path.exists(data_path):
        print("Data not found. Please run fetch_data.py and feature_engineer.py first.")
        return
        
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Time-based splitting
    # Total ~30 days. 
    # Train: 70%, Val: 15%, Test: 15%
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    
    print(f"Dataset Split -> Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # SPY Benchmark is roughly 10% annualized. We aim to beat 15% to be safe.
    target_annual_roi = 15.0 
    
    # Grid Search Space
    grid = [
        {'seq_len': 30, 'hidden_dim': 64, 'epochs': 10, 'lr': 1e-3, 'batch': 128},
        {'seq_len': 60, 'hidden_dim': 128, 'epochs': 15, 'lr': 1e-3, 'batch': 256},
        {'seq_len': 60, 'hidden_dim': 64, 'epochs': 25, 'lr': 5e-4, 'batch': 64},
        {'seq_len': 120, 'hidden_dim': 128, 'epochs': 15, 'lr': 1e-3, 'batch': 256},
    ]
    
    best_test_roi = -float('inf')
    best_config = None
    
    for i, config in enumerate(grid):
        print(f"\n[Test {i+1}/{len(grid)}] Training with config: {config}")
        
        # 1. Dataset setup for this specific seq_len
        dataset = TimeSeriesDataset(data_path, seq_length=config['seq_len'])
        
        # Slicing the PyTorch dataset to match train limit
        train_indices = list(range(train_end - config['seq_len']))
        train_sampler = torch.utils.data.SequentialSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=config['batch'], sampler=train_sampler, drop_last=True, pin_memory=True
        )
        
        # 2. Model Init
        input_dim = len(dataset.feature_cols)
        model = LSTMTradingModel(
            input_dim=input_dim, 
            hidden_dim=config['hidden_dim'], 
            num_layers=2, 
            output_dim=2, 
            dropout=0.3
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        # 3. Train Loop
        model.train()
        for epoch in range(config['epochs']):
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
        # 4. Validate
        val_roi, val_annual, vw, vl = calculate_roi(df_val, model, config['seq_len'], device)
        print(f"Validation Results -> ROI: {val_roi:.2f}% | Annualized: {val_annual:.2f}% | W/L: {vw}/{vl}")
        
        # Check against SPY benchmark
        if val_annual > target_annual_roi:
            print(f"  -> SUCCESS! Validation beat SPY ({val_annual:.2f}% > {target_annual_roi}%). Testing on Unseen Data...")
            test_roi, test_annual, tw, tl = calculate_roi(df_test, model, config['seq_len'], device)
            print(f"Test Results (UNSEEN DATA) -> ROI: {test_roi:.2f}% | Annualized: {test_annual:.2f}% | W/L: {tw}/{tl}")
            
            if test_annual > target_annual_roi:
                print("\n" + "="*50)
                print(f"HOLY GRAIL FOUND! Configuration {i+1} is profitable on unseen test data.")
                print(f"Test ROI: {test_roi:.2f}% | Annualized Test ROI: {test_annual:.2f}%")
                best_model_path = f"models/opt_model_conf{i+1}.pth"
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print(f"Model strictly saved at: {best_model_path}")
                print("="*50)
                return
            else:
                 print("  -> FAILED on Unseen Test Data. It was overfit. Rinsing and repeating...")
        else:
            print(f"  -> FAILED Validation. Didn't beat SPY ({val_annual:.2f}%). Rinsing and repeating...")
            
    print("\n[Search Concluded] The grid search exhausted without finding a Holy Grail configuration beating SPY on both Val and Test data.")
    print("Recommendation: Widen the search grid, engineer more features, or increase data fetch (e.g. 90 days).")

if __name__ == "__main__":
    optimize_bot()
