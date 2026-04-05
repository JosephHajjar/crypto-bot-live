import os, sys, json, time
import torch
import numpy as np
import pandas as pd
import requests

from data.feature_engineer_btc import compute_live_features, get_feature_cols
from ml.model import AttentionLSTMModel

# Triple barrier params (match training)
TP = 0.015
SL = 0.0075
MAX_BARS = 16
FEE = 0.001

def fetch_recent_15m(days):
    all_raw = []
    end_ts = int(time.time() * 1000)
    ims = 900000  # 15m in ms
    candles_needed = days * 96  # 96 candles per day
    pages = (candles_needed + 999) // 1000

    for page in range(pages):
        page_end = end_ts - (page * 1000 * ims)
        page_start = page_end - (1000 * ims)
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=1000&startTime={page_start}&endTime={page_end}"
        res = requests.get(url, timeout=15)
        data = res.json()
        if data and isinstance(data, list):
            all_raw = data + all_raw
        time.sleep(0.1)

    seen = set()
    unique = []
    for r in all_raw:
        if r[0] not in seen:
            seen.add(r[0])
            unique.append(r)
    unique.sort(key=lambda x: x[0])

    df = pd.DataFrame(unique, columns=['timestamp','open','high','low','close','volume',
                                        'close_time','qav','num_trades','tbbav','tbqav','ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df

def backtest_with_threshold(signals, close, high, low, seq_len, long_threshold):
    capital = 10000.0
    initial = capital
    max_cap = capital
    max_dd = 0.0
    trades = []
    i = 0

    while i < len(signals):
        idx = i + seq_len
        if idx + MAX_BARS >= len(close):
            break
        if capital > 0:
            prob = signals[i]
            if prob > long_threshold:
                # LONG
                entry = close[idx]
                entry_cap = capital
                tp_price = entry * (1 + TP)
                sl_price = entry * (1 - SL)
                exit_p = None
                reason = None
                for j in range(1, MAX_BARS + 1):
                    jdx = idx + j
                    if jdx >= len(close): break
                    if low[jdx] <= sl_price:
                        exit_p = sl_price; reason = 'SL'; break
                    if high[jdx] >= tp_price:
                        exit_p = tp_price; reason = 'TP'; break
                if exit_p is None:
                    exit_p = close[idx + MAX_BARS]; reason = 'TIME'
                ret = (exit_p - entry) / entry - FEE * 2
                capital = entry_cap * (1 + ret)
                if capital > max_cap: max_cap = capital
                dd = (capital - max_cap) / max_cap * 100
                if dd < max_dd: max_dd = dd
                trades.append({'win': ret > 0, 'ret': ret, 'reason': reason, 'type': 'LONG'})
                i += MAX_BARS
            else:
                i += 1
        else:
            i += 1

    roi = (capital - initial) / initial * 100
    wins = sum(1 for t in trades if t['win'])
    losses = len(trades) - wins
    longs = sum(1 for t in trades if t['type'] == 'LONG')
    
    if len(trades) > 1:
        rets = [t['ret'] for t in trades]
        sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(len(trades))
    else:
        sharpe = 0

    return {
        'roi': roi, 'sharpe': sharpe, 'max_dd': max_dd,
        'trades': len(trades), 'wins': wins, 'losses': losses,
        'longs': longs
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler_path = 'data_storage/BTC_USDT_15m_scaler.json'
    config_path = 'models/trial_255_config.json'
    model_path = 'models/trial_255.pth'

    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    print(f"Loading Model 255 onto {device}...")
    model = AttentionLSTMModel(
        input_dim=cfg['input_dim'],
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        output_dim=2,
        dropout=cfg['dropout'],
        num_heads=cfg['num_heads']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    seq_len = cfg['seq_len']

    max_days = 90 + 30 # 30 days buffer
    print(f"\nFetching {max_days} days of 15m BTC data...")
    df_raw = fetch_recent_15m(max_days)
    
    print("Computing features...")
    df_feat = compute_live_features(df_raw, scaler_path)
    
    # Take latest 90 days for backtesting explicitly
    candles = 90 * 96
    df_slice = df_feat.iloc[-candles:] if len(df_feat) > candles else df_feat
    
    feature_cols = get_feature_cols()
    
    available = [c for c in feature_cols if c in df_slice.columns]
    X = df_slice[available].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Use actual full dataframe columns matching index
    close = df_feat['close'].values if 'close' in df_feat.columns else df_raw['close'].values[-len(X):]
    high = df_feat['high'].values if 'high' in df_feat.columns else df_raw['high'].values[-len(X):]
    low = df_feat['low'].values if 'low' in df_feat.columns else df_raw['low'].values[-len(X):]

    # Align close size to X size
    close = close[-len(X):]
    high = high[-len(X):]
    low = low[-len(X):]

    # Precompute signals (since model output is constant across different thresholds)
    print("Running model inference to precompute all probabilities...")
    all_seqs = np.array([X[i:i+seq_len] for i in range(len(X) - seq_len)])
    seq_tensor = torch.tensor(all_seqs, dtype=torch.float32).to(device)
    
    signals = []
    with torch.no_grad():
        for i in range(0, len(seq_tensor), 2048):
            batch = seq_tensor[i:i+2048]
            out = model(batch)
            probs = torch.softmax(out, dim=1)
            bull_probs = probs[:, 1].cpu().tolist()
            signals.extend(bull_probs)
            
    # Long Thresholds to test: 0.50 (Baseline), 0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 0.995, 0.999]
    
    results = []
    print("\n" + "="*85)
    print(f"{'Long Threshold':<16} | {'Confidence':<12} | {'ROI':<9} | {'Sharpe':<8} | {'Trades':<8} | {'Longs':<8} | {'Win/Loss':<11}")
    print("="*85)

    for threshold in thresholds:
        res = backtest_with_threshold(signals, close, high, low, seq_len, threshold)
        
        conf_str = f"{threshold*100:.1f}% BULL"
        thresh_str = str(threshold)
        ls_str = f"{res['longs']}"
        wl_str = f"{res['wins']}/{res['losses']}"
        
        print(f"{thresh_str:<16} | {conf_str:<12} | {res['roi']:>8.2f}% | {res['sharpe']:>6.2f}   | {res['trades']:<8} | {ls_str:<8} | {wl_str:<11}")
        
        results.append({
            'threshold': threshold,
            'roi': res['roi'],
            'trades': res['trades']
        })

    best_thresh = max(results, key=lambda x: x['roi'])
    print("="*85)
    print(f"\nBest Performing Parameter: Threshold {best_thresh['threshold']} (ROI: {best_thresh['roi']:.2f}%)")

if __name__ == '__main__':
    main()
