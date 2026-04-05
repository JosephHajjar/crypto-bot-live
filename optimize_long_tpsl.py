import os, sys, json, time
import torch
import numpy as np
import pandas as pd
import requests
import optuna

from data.feature_engineer import compute_live_features, get_feature_cols
from ml.model import AttentionLSTMModel

# Fixed threshold from earlier threshold optimization
LONG_THRESHOLD = 0.60
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

def backtest_tpsl(signals, close, high, low, seq_len, tp, sl, max_bars):
    capital = 10000.0
    initial = capital
    max_cap = capital
    max_dd = 0.0
    trades = []
    i = 0

    while i < len(signals):
        idx = i + seq_len
        if idx + max_bars >= len(close):
            break
        if capital > 0:
            prob = signals[i]
            if prob > LONG_THRESHOLD:
                # LONG
                entry = close[idx]
                entry_cap = capital
                tp_price = entry * (1 + tp)
                sl_price = entry * (1 - sl)
                exit_p = None
                reason = None
                for j in range(1, max_bars + 1):
                    jdx = idx + j
                    if jdx >= len(close): break
                    if low[jdx] <= sl_price:
                        exit_p = sl_price; reason = 'SL'; break
                    if high[jdx] >= tp_price:
                        exit_p = tp_price; reason = 'TP'; break
                if exit_p is None:
                    exit_p = close[idx + max_bars]; reason = 'TIME'
                    
                ret = (exit_p - entry) / entry - FEE * 2
                capital = entry_cap * (1 + ret)
                if capital > max_cap: max_cap = capital
                dd = (capital - max_cap) / max_cap * 100
                if dd < max_dd: max_dd = dd
                trades.append({'win': ret > 0, 'ret': ret})
                i += max_bars
            else:
                i += 1
        else:
            i += 1

    roi = (capital - initial) / initial * 100
    wins = sum(1 for t in trades if t['win'])
    losses = len(trades) - wins
    
    if len(trades) > 1:
        rets = [t['ret'] for t in trades]
        sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(len(trades))
    else:
        sharpe = 0

    return {
        'roi': roi, 'sharpe': sharpe, 'max_dd': max_dd,
        'trades': len(trades), 'wins': wins, 'losses': losses
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

    # Precompute signals ONCE!
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
            
    print("Precomputation complete! Starting hyperparameter search for TP / SL / Max Bars...\n")
    
    def objective(trial):
        # Suggest structural parameters
        tp = trial.suggest_float('take_profit', 0.005, 0.05, step=0.0025)
        sl = trial.suggest_float('stop_loss', 0.005, 0.05, step=0.0025)
        max_bars = trial.suggest_categorical('max_hold_bars', [8, 12, 16, 24, 32, 48, 64])
        
        # Simulate over the EXACT same fixed prediction array (takes milliseconds!)
        res = backtest_tpsl(signals, close, high, low, seq_len, tp=tp, sl=sl, max_bars=max_bars)
        
        # Constraints
        if res['trades'] < 30:
            return -100.0
        
        if res['sharpe'] <= 0:
            return -100.0
            
        return res['roi']

    # Silence verbose Optuna logs because it runs thousands of trials a second
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=5000, show_progress_bar=True)
    
    best = study.best_trial
    best_res = backtest_tpsl(signals, close, high, low, seq_len, best.params['take_profit'], best.params['stop_loss'], best.params['max_hold_bars'])
    
    print("\n" + "="*60)
    print("BEST TAKE PROFIT / STOP LOSS CONFIGURATION FOUND")
    print("="*60)
    print(f"Take Profit:   {best.params['take_profit']*100:>5.2f}%")
    print(f"Stop Loss:     {best.params['stop_loss']*100:>5.2f}%")
    print(f"Max Hold Bars: {best.params['max_hold_bars']}")
    print("-" * 60)
    print(f"ROI (90 Days): {best_res['roi']:.2f}%")
    print(f"Sharpe Ratio:  {best_res['sharpe']:.2f}")
    print(f"Trades:        {best_res['trades']}")
    print(f"Win Rate:      {(best_res['wins']/best_res['trades']*100):.1f}% ({best_res['wins']}W/{best_res['losses']}L)")
    print("="*60)

if __name__ == "__main__":
    main()
