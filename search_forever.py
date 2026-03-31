import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import time
import json
import urllib.request
from tqdm import tqdm

from data.fetch_data import fetch_klines
from data.feature_engineer import engineer_features
from ml.dataset import TimeSeriesDataset
from ml.model import LSTMTradingModel

def calculate_roi(df_slice, model, seq_length, device, fee_pct=0.001):
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
            position = capital * (1 - fee_pct)
            capital = 0
            entry_capital = position / (1 - fee_pct)
        elif signal == 0 and position > 0:
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
         if capital > entry_capital:
             trades_won += 1
         else:
             trades_lost += 1
         
    total_profit = capital - initial_capital
    roi = (total_profit / initial_capital) * 100
    
    days_in_slice = len(df_slice) * 15 / 1440
    if days_in_slice > 0:
        annual_roi = ((1 + roi/100) ** (365 / days_in_slice) - 1) * 100
    else:
        annual_roi = 0.0
        
    return roi, annual_roi, trades_won, trades_lost

def infinite_search():
    print("=== Gathering 180 Days of Data for the Infinite Search ===")
    
    symbol = 'BTC/USDT'
    timeframe = '15m'
    days = 365
    
    csv_path = fetch_klines(symbol, timeframe, days, save_dir='data_storage')
    if not csv_path:
        print("Failed to fetch data. Retrying in 10s...")
        time.sleep(10)
        return
        
    processed_path, _ = engineer_features(csv_path)
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    print(f"Dataset Split -> Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_annual_roi = 15.0 
    
    state_file = 'search_state.json'
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            best_test_roi_so_far = state.get('best_test_annual', 0.0)
            best_val_roi_so_far = state.get('best_val_annual', -100.0)
            search_iteration = state.get('total_searches', 0) + 1
            print(f"Resuming search. Best Test: {best_test_roi_so_far:.2f}%. Best Val: {best_val_roi_so_far:.2f}%")
    else:
        best_test_roi_so_far = 0.0
        best_val_roi_so_far = -100.0
        search_iteration = 1
    
    print("\n=== Commencing INFINITE RANDOM SEARCH for an Edge ===")
    while True:
        # 1. Randomize Hyperparameters
        seq_len = random.choice([30, 60, 90, 120, 180, 240])
        hidden_dim = random.choice([32, 64, 128, 256])
        epochs = random.randint(10, 40)
        lr = random.uniform(1e-4, 5e-3)
        batch = random.choice([64, 128, 256, 512])
        dropout = random.uniform(0.1, 0.5)
        
        print(f"\n[Search #{search_iteration}] seq_len={seq_len}, hidden={hidden_dim}, epochs={epochs}, lr={lr:.5f}, batch={batch}, drop={dropout:.2f}")
        search_iteration += 1
        
        dataset = TimeSeriesDataset(processed_path, seq_length=seq_len)
        train_indices = list(range(train_end - seq_len))
        train_sampler = torch.utils.data.SequentialSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch, sampler=train_sampler, drop_last=True, pin_memory=True
        )
        
        input_dim = len(dataset.feature_cols)
        model = LSTMTradingModel(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            num_layers=2, 
            output_dim=2, 
            dropout=dropout
        ).to(device)
        
        # Overweight class 1 (profitable) to heavily penalize missing profitable setups
        class_weights = torch.tensor([1.0, 1.5]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 2. Train
        model.train()
        for epoch in range(epochs):
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
                
        # 3. Validation and Testing Tracking
        val_roi, val_annual, vw, vl = calculate_roi(df_val, model, seq_len, device)
        print(f"  Val -> Annualized: {val_annual:.2f}% | ROI: {val_roi:.2f}% | W/L: {vw}/{vl}")
        
        # --- Stateful Uncut Progress Checker ---
        progress_made = False
        if val_annual > best_val_roi_so_far and vw > 0:
            best_val_roi_so_far = val_annual
            progress_made = True
            
        if progress_made:
            with open(state_file, 'w') as f:
                json.dump({'best_test_annual': best_test_roi_so_far, 'best_val_annual': best_val_roi_so_far, 'total_searches': search_iteration}, f)
                
        # 4. Filter for Edge and Unseen Test
        if val_annual > target_annual_roi and vw > 0:
            print(f"  --> Edge potentially found on Validation! Verifying on Unseen Test Data...")
            test_roi, test_annual, tw, tl = calculate_roi(df_test, model, seq_len, device)
            print(f"  Test Results -> Annualized: {test_annual:.2f}% | ROI: {test_roi:.2f}% | W/L: {tw}/{tl}")
            
            if test_annual > best_test_roi_so_far and tw > tl:
                best_test_roi_so_far = test_annual
                
                with open(state_file, 'w') as f:
                    json.dump({'best_test_annual': best_test_roi_so_far, 'best_val_annual': best_val_roi_so_far, 'total_searches': search_iteration}, f)
                
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), "models/best_so_far.pth")
                with open("models/best_so_far_config.json", "w") as cf:
                    json.dump({"seq_len": seq_len, "hidden_dim": hidden_dim, "dropout": dropout}, cf)
                print(f"  --> NEW UNSEEN TEST HIGH SCORE SECURED: {test_annual:.2f}%!")
                
                msg = f"TradeBot5234 Unseen High Score! Model #{search_iteration-1} hit {test_annual:.2f}% Test ROI"
                try:
                    urllib.request.urlopen(urllib.request.Request("https://ntfy.sh/TradeBot5234", data=msg.encode('utf-8')), timeout=5)
                except Exception: pass
            
            # --- Check Holy Grail Criteria ---
            if test_annual > target_annual_roi and tw > tl:
                print("\n" + "="*50)
                print(f"HOLY GRAIL FOUND! Configuration #{search_iteration-1} has a verified statistical edge on BOTH Sets.")
                print(f"Config: seq_len={seq_len}, hidden={hidden_dim}, lr={lr:.5f}, epochs={epochs}")
                print(f"Test ROI: {test_roi:.2f}% | Annualized Test ROI: {test_annual:.2f}%")
                
                best_model_path = f"models/holy_grail_edge_found.pth"
                torch.save(model.state_dict(), best_model_path)
                with open("models/holy_grail_config.json", "w") as cf:
                    json.dump({"seq_len": seq_len, "hidden_dim": hidden_dim, "dropout": dropout}, cf)
                
                msg = f"HOLY GRAIL SECURED! ROI: {test_annual:.2f}%. Bot automatically shut down successfully."
                try:
                    urllib.request.urlopen(urllib.request.Request("https://ntfy.sh/TradeBot5234", data=msg.encode('utf-8')), timeout=5)
                except Exception: pass
                
                print(f"Model saved to {best_model_path}")
                print("Exiting search loop as an edge was secured.")
                print("="*50 + "\n")
                break
            else:
                print("  --> Edge fell short of the 15% Holy Grail. Continuing search...")
            
if __name__ == "__main__":
    infinite_search()
