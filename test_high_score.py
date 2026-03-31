import os
import json
import torch
import pandas as pd
import numpy as np
from ml.model import LSTMTradingModel

def test_high_score():
    print("=== Loading +12.42% High Score Architecture ===")
    
    config_path = 'models/best_config.json'
    model_path = 'models/best_so_far.pth'
    data_path = 'data_storage/BTC_USDT_15m_processed.csv'
    
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print("Could not find best_so_far.pth! Waiting...")
        return
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    seq_len = config['seq_len']
    hidden_dim = config['hidden_dim']
    dropout = config.get('dropout', 0.2)
    print(f"Loaded Identity Matrix: seq_len={seq_len}, hidden={hidden_dim}, drop={dropout}")
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    # Validation constraint exactly matching search_forever.py
    df_val = df.iloc[train_end:val_end]
    print(f"Mapping Test Array size: {len(df_val)} lines (roughly exactly 54 Days)")
    
    feature_cols = [
        'Returns', 'SMA_10', 'SMA_50', 'RSI_14', 
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
        'ATR_14', 'Vol_Ratio'
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMTradingModel(
        input_dim=len(feature_cols), 
        hidden_dim=hidden_dim, 
        num_layers=2, 
        output_dim=2, 
        dropout=dropout
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    fee_pct = 0.001
    X_val = df_val[feature_cols].values
    close_prices = df_val['close'].values
    timestamps = df_val.index.values
    
    capital = 10000.0
    initial_capital = capital
    position = 0 
    entry_capital = 0
    trades_won = 0
    trades_lost = 0
    
    all_seqs = [X_val[i-seq_len : i] for i in range(seq_len, len(X_val) - 1)]
    seq_tensor = torch.tensor(np.array(all_seqs), dtype=torch.float32).to(device)
    
    signals = []
    with torch.no_grad():
        batch_size = 2048
        for i in range(0, len(seq_tensor), batch_size):
            batch = seq_tensor[i : i+batch_size]
            out = model(batch)
            _, pred = torch.max(out, 1)
            signals.extend(pred.cpu().tolist())
            
    trade_history = []
            
    for idx, i in enumerate(range(seq_len, len(X_val) - 1)):
        signal = signals[idx]
        current_return = (close_prices[i+1] - close_prices[i]) / close_prices[i]
        timestamp_str = pd.to_datetime(timestamps[i]).strftime('%Y-%m-%d %H:%M')
        
        if signal == 1 and position == 0:
            position = capital * (1 - fee_pct)
            capital = 0
            entry_capital = position / (1 - fee_pct)
            trade_history.append({'Type': 'BUY', 'Time': timestamp_str, 'Price': round(close_prices[i], 2), 'Bal': round(entry_capital, 2)})
            
        elif signal == 0 and position > 0:
            capital = position * (1 - fee_pct)
            win = capital > entry_capital
            if win: trades_won += 1
            else: trades_lost += 1
            position = 0
            trade_history.append({'Type': 'SELL', 'Time': timestamp_str, 'Price': round(close_prices[i], 2), 'Bal': round(capital, 2), 'Win': win})
            
        if position > 0:
            position = position * (1 + current_return)
                
    if position > 0:
         capital = position * (1 - fee_pct)
         win = capital > entry_capital
         if win: trades_won += 1
         else: trades_lost += 1
         position = 0
         timestamp_str = pd.to_datetime(timestamps[len(X_val)-1]).strftime('%Y-%m-%d %H:%M')
         trade_history.append({'Type': 'LIQ', 'Time': timestamp_str, 'Price': round(close_prices[len(X_val)-1], 2), 'Bal': round(capital, 2), 'Win': win})
         
    total_profit = capital - initial_capital
    roi = (total_profit / initial_capital) * 100
    days_in_slice = len(df_val) * 15 / 1440
    annual_roi = ((1 + roi/100) ** (365 / days_in_slice) - 1) * 100
    
    print("\n--- VISUAL TRADE LOG ---")
    for t in trade_history:
        if t['Type'] == 'BUY':
             print(f"🟢 [BUY]  {t['Time']} | BTC @ ${t['Price']} | Account=$10,000")
        else:
            win_str = "WIN " if t['Win'] else "LOSS"
            print(f"🔴 [{t['Type']}]  {t['Time']} | BTC @ ${t['Price']} | Account=${t['Bal']} | {win_str}")
            print("-" * 50)
            
    print(f"\nFinal State -> Final ROI: {annual_roi:.2f}% | Clean Trades (W/L): {trades_won}/{trades_lost}")

if __name__ == "__main__":
    test_high_score()
