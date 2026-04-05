import sys, os, json, torch
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
from ml.model import AttentionLSTMModel
from data.feature_engineer import get_feature_cols

def run_backtest(df_slice, model, seq_length, device, mode):
    # mode: 'LONG_ONLY' or 'LONG_SHORT'
    feature_cols = get_feature_cols()
    available = [c for c in feature_cols if c in df_slice.columns]
    
    X = df_slice[available].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    close = df_slice['close'].values
    high = df_slice['high'].values
    low = df_slice['low'].values
    
    max_bars = 16
    tp = 0.015
    sl = 0.0075
    fee_pct = 0.001
    
    if len(X) < seq_length + max_bars + 10:
        return 0, 0
    
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
    
    capital = 10000.0
    initial_capital = capital
    i = 0
    trades_count = 0
    
    while i < len(signals):
        sig_idx = i + seq_length
        if sig_idx + max_bars >= len(close): break
        
        if capital > 0:
            prob = signals[i]
            
            if prob > 0.50:
                # LONG
                entry = close[sig_idx]
                entry_cap = capital
                tp_p = entry * (1 + tp)
                sl_p = entry * (1 - sl)
                ex = None
                for j in range(1, max_bars + 1):
                    idx = sig_idx + j
                    if idx >= len(close): break
                    if low[idx] <= sl_p: ex = sl_p; break
                    if high[idx] >= tp_p: ex = tp_p; break
                if ex is None: ex = close[sig_idx + max_bars]
                
                ret = (ex - entry) / entry - fee_pct * 2
                capital = entry_cap * (1 + ret)
                trades_count += 1
                i += max_bars
                
            elif prob < 0.45 and mode == 'LONG_SHORT':
                # SHORT
                entry = close[sig_idx]
                entry_cap = capital
                tp_p = entry * (1 - tp)
                sl_p = entry * (1 + sl)
                ex = None
                for j in range(1, max_bars + 1):
                    idx = sig_idx + j
                    if idx >= len(close): break
                    if high[idx] >= sl_p: ex = sl_p; break
                    if low[idx] <= tp_p: ex = tp_p; break
                if ex is None: ex = close[sig_idx + max_bars]
                
                ret = (entry - ex) / entry - fee_pct * 2
                capital = entry_cap * (1 + ret)
                trades_count += 1
                i += max_bars
            else:
                i += 1
        else:
            i += 1
            
    roi = (capital - initial_capital) / initial_capital * 100
    return roi, trades_count

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv("data_storage/BTC_USDT_15m_processed.csv", index_col=0, parse_dates=True)
    
    cfg_path = "models/trial_255_config.json"
    weights_path = "models/trial_255.pth"
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
        
    model = AttentionLSTMModel(
        input_dim=cfg['input_dim'],
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        output_dim=2,
        dropout=cfg['dropout'],
        num_heads=cfg['num_heads']
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    
    periods = [1, 3, 7, 14, 30, 60, 90]
    
    print("\n=== MODEL 255 BACKTEST COMPARISON ===")
    print(f"{'PERIOD':<10} | {'LONG ONLY ROI':<15} | {'LONG+SHORT ROI':<15} | {'TRADES (LO/LS)'}")
    print("-" * 65)
    
    for dys in periods:
        bars = dys * 96
        start_idx = len(df) - bars - cfg['seq_len']
        if start_idx < 0: start_idx = 0
        df_slice = df.iloc[start_idx:]
        
        roi_lo, t_lo = run_backtest(df_slice, model, cfg['seq_len'], device, 'LONG_ONLY')
        roi_ls, t_ls = run_backtest(df_slice, model, cfg['seq_len'], device, 'LONG_SHORT')
        
        print(f"{dys:>2} Days   | {roi_lo:>8.2f}%       | {roi_ls:>8.2f}%       | {t_lo}/{t_ls}")

if __name__ == "__main__":
    main()
