import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from ml.model import load_model, LSTMTradingModel

def backtest_model(data_path, model_path, seq_length=60, initial_capital=10000.0, fee_pct=0.001):
    """
    Simulates trading based on model predictions over the historic (validation) dataset.
    """
    print(f"Loading data for backtesting from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Needs to match exactly the features used in dataset.py
    feature_cols = [
        'Returns', 'SMA_10', 'SMA_50', 'RSI_14', 
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
        'ATR_14', 'Vol_Ratio'
    ]
    
    # We will out-of-sample backtest ONLY on the last 20% of the data
    val_split_idx = int(len(df) * 0.8)
    df_val = df.iloc[val_split_idx:].copy()
    
    print(f"Backtesting on {len(df_val)} periods...")
    
    # Note: To backtest sequentially using LSTM we must build the tensors
    X_val = df_val[feature_cols].values
    close_prices = df_val['close'].values if 'close' in df_val.columns else np.random.randn(len(df_val)) # Dummy if we dropped original closing prices
    returns = df_val['Returns'].values
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMTradingModel(input_dim=len(feature_cols), hidden_dim=128, num_layers=2, output_dim=2)
    model = load_model(model, filepath=model_path, device=device)
    model.eval()

    # Backtesting State
    capital = initial_capital
    position = 0 # 0 for flat, 1 for long
    equity_curve = []
    trades_won = 0
    trades_lost = 0
    
    print("Running vector predictions and simulating execution...")
    
    # Since we need sequence length context, we can only start trading after seq_length periods
    length = len(X_val) - 1 - seq_length
    if length <= 0: return {}
    
    print("Vectorizing Inference...")
    all_seqs = [X_val[i-seq_length : i] for i in range(seq_length, len(X_val) - 1)]
    seq_tensor = torch.tensor(np.array(all_seqs), dtype=torch.float32).to(device)
    
    signals = []
    with torch.no_grad():
        batch_size = 2048
        for i in range(0, len(seq_tensor), batch_size):
            batch = seq_tensor[i : i+batch_size]
            out = model(batch)
            _, pred = torch.max(out, 1)
            signals.extend(pred.cpu().tolist())
            
    print("Simulating Execution...")
    for idx, i in enumerate(tqdm(range(seq_length, len(X_val) - 1))):
        signal = signals[idx] # 1 = UP, 0 = NOT UP
        
        # 3. Execute Trading Logic
        
        # Fix: We must determine actual asset return from raw close prices, NOT Z-scored features
        current_return = (close_prices[i+1] - close_prices[i]) / close_prices[i]
        
        if signal == 1 and position == 0:
            # Buy
            position = capital * (1 - fee_pct) # Buy full account worth, pay fee
            capital = 0
            entry_capital = position / (1 - fee_pct) # Effectively starting capital for trade
        elif signal == 0 and position > 0:
            # Sell
            capital = position * (1 - fee_pct)
            
            # Check if profit
            if capital > entry_capital:
                 trades_won += 1
            else:
                 trades_lost += 1
                 
            position = 0
        
        # Update equity
        current_equity = capital if position == 0 else position * (1 + current_return)
        equity_curve.append(current_equity)
        
        # Update position value if held
        if position > 0:
            position = position * (1 + current_return)
                
    # Final close out
    if position > 0:
         capital = position * (1 - fee_pct)
         
    total_profit = capital - initial_capital
    roi = (total_profit / initial_capital) * 100
    win_rate = trades_won / (trades_won + trades_lost) if (trades_won + trades_lost) > 0 else 0
    
    print(f"\n--- Backtest Results ---")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital:   ${capital:.2f}")
    print(f"Total Profit:    ${total_profit:.2f}")
    print(f"ROI:             {roi:.2f}%")
    print(f"Win Rate:        {win_rate*100:.2f}% ({trades_won}W / {trades_lost}L)")
    
    return {
        'Final_Capital': capital,
        'ROI': roi,
        'Win_Rate': win_rate
    }
    
if __name__ == "__main__":
    pass
