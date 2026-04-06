import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import optuna

from ml.model import AttentionLSTMModel
from data.feature_engineer_btc import get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def precompute_probs():
    print("Pre-computing model probabilities once for fast grid search...")
    df = pd.read_csv('data_storage/BTC_USDT_15m_processed.csv')
    df = df.iloc[-2880:].copy() # Last 30 days (4/hr * 24hr * 30)
    
    with open('models/holy_grail_config.json', 'r') as f:
        cfg_long = json.load(f)
    s_long = cfg_long.get('seq_len', 128)
    
    model_long = AttentionLSTMModel(
        input_dim=cfg_long['input_dim'], hidden_dim=cfg_long['hidden_dim'],
        num_layers=cfg_long['num_layers'], output_dim=2, dropout=cfg_long['dropout'], num_heads=cfg_long['num_heads']
    ).to(device)
    model_long.load_state_dict(torch.load('models/holy_grail.pth', map_location=device, weights_only=True))
    model_long.eval()

    with open('models_short/holy_grail_short_config.json', 'r') as f:
        cfg_short = json.load(f)
    s_short = cfg_short.get('seq_len', 128)
    
    model_short = AttentionLSTMModel(
        input_dim=cfg_short['input_dim'], hidden_dim=cfg_short['hidden_dim'],
        num_layers=cfg_short['num_layers'], output_dim=2, dropout=cfg_short['dropout'], num_heads=cfg_short['num_heads']
    ).to(device)
    model_short.load_state_dict(torch.load('models_short/holy_grail_short.pth', map_location=device, weights_only=True))
    model_short.eval()

    MAX_SEQ_LEN = max(s_long, s_short)
    feature_cols = get_feature_cols()
    
    unix_times = df['timestamp'].values
    close_prices = df['close'].values
    features_np = df[feature_cols].values.astype(np.float32)
    
    batch_size = 512
    all_bull = []
    all_bear = []
    
    with torch.no_grad():
        for start in range(0, len(features_np) - MAX_SEQ_LEN + 1, batch_size):
            end = min(start + batch_size, len(features_np) - MAX_SEQ_LEN + 1)
            
            batch_l = [features_np[i + MAX_SEQ_LEN - s_long : i + MAX_SEQ_LEN] for i in range(start, end)]
            batch_s = [features_np[i + MAX_SEQ_LEN - s_short : i + MAX_SEQ_LEN] for i in range(start, end)]
            
            t_long = torch.tensor(np.array(batch_l)).to(device)
            t_short = torch.tensor(np.array(batch_s)).to(device)
            
            p_bull = torch.softmax(model_long(t_long), dim=1)[:, 1].cpu().numpy().tolist()
            p_bear = torch.softmax(model_short(t_short), dim=1)[:, 1].cpu().numpy().tolist()
            
            all_bull.extend(p_bull)
            all_bear.extend(p_bear)
            
    return close_prices[MAX_SEQ_LEN - 1:], np.array(all_bull), np.array(all_bear)

# Global for fast optuna
PRECOMPUTED_CLOSES = None
PRECOMPUTED_BULL = None
PRECOMPUTED_BEAR = None

def evaluate_sim(close_prices, probs_bull, probs_bear, enter_margin, flip_margin, flat_margin, enable_flat=True):
    position = 'flat'
    entry_price = 0
    total_pnl = 0
    trade_count = 0
    
    for i in range(len(probs_bull)):
        c_p = close_prices[i]
        bull = probs_bull[i]
        bear = probs_bear[i]
        
        diff_bull = bull - bear
        diff_bear = bear - bull
        
        if position == 'flat':
            if diff_bull >= enter_margin:
                position = 'long'
                entry_price = c_p
            elif diff_bear >= enter_margin:
                position = 'short'
                entry_price = c_p
        elif position == 'long':
            # Check if we should flip directly
            if diff_bear >= flip_margin:
                total_pnl += (c_p - entry_price) / entry_price * 100 - 0.07 # Assume 0.07% fees
                trade_count += 1
                
                position = 'short'
                entry_price = c_p
                continue
            
            # Check if we should go flat
            if enable_flat and diff_bull < flat_margin:
                total_pnl += (c_p - entry_price) / entry_price * 100 - 0.07
                trade_count += 1
                position = 'flat'
                entry_price = 0
                
        elif position == 'short':
             # Check if we should flip directly
            if diff_bull >= flip_margin:
                total_pnl += (entry_price - c_p) / entry_price * 100 - 0.07
                trade_count += 1
                
                position = 'long'
                entry_price = c_p
                continue
            
            # Check if we should go flat
            if enable_flat and diff_bear < flat_margin:
                total_pnl += (entry_price - c_p) / entry_price * 100 - 0.07
                trade_count += 1
                position = 'flat'
                entry_price = 0
                
    return total_pnl, trade_count

def objective(trial):
    # Proportional logic tuning
    # Strategy parameters:
    # enter_margin: margin difference required to enter a trade from a flat state (0.0 to 0.4)
    # flip_margin: margin difference required to completely flip short <-> long (0.0 to 0.4)
    # flat_margin: if the ruling edge drops below this lead, we close out and go flat
    # enable_flat: boolean, should we ever go to cash? Or ALWAYS stay in market.

    enable_flat = trial.suggest_categorical('enable_flat', [True, False])
    
    enter_margin = trial.suggest_float('enter_margin', -0.1, 0.4)
    flip_margin = trial.suggest_float('flip_margin', -0.1, 0.4)
    
    if enable_flat:
        flat_margin = trial.suggest_float('flat_margin', -0.1, 0.3)
    else:
        flat_margin = -999.0 # disabled
        
    pnl, trades = evaluate_sim(PRECOMPUTED_CLOSES, PRECOMPUTED_BULL, PRECOMPUTED_BEAR, enter_margin, flip_margin, flat_margin, enable_flat=enable_flat)
    
    # Penalize too few trades to ensure statistical significance
    if trades < 30:
        return -999.0
        
    return pnl

if __name__ == '__main__':
    print("Starting optimization...")
    closes, bull, bear = precompute_probs()
    PRECOMPUTED_CLOSES = closes
    PRECOMPUTED_BULL = bull
    PRECOMPUTED_BEAR = bear
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5000, show_progress_bar=True)
    
    print("\n\n------------- BEST PARAMS -------------")
    print(study.best_params)
    print(f"Best ROI (with fees): {study.best_value:.2f}%")
