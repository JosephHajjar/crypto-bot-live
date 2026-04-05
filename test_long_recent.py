import os, sys, json, time
import torch
import numpy as np
import pandas as pd

bot_dir = 'c:\\Users\\asdf\\.gemini\\antigravity\\scratch\\ml_trading_bot'
sys.path.append(bot_dir)

from optimize_long_tpsl import fetch_recent_15m, backtest_tpsl
from data.feature_engineer_btc import compute_live_features, get_feature_cols
from ml.model import AttentionLSTMModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler_path = os.path.join(bot_dir, 'data_storage/BTC_USDT_15m_scaler.json')
config_path = os.path.join(bot_dir, 'models/trial_255_config.json')
model_path = os.path.join(bot_dir, 'models/trial_255.pth')

with open(config_path, 'r') as f:
    cfg = json.load(f)

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

# Fetch recent 10 days to have enough buffer for slow moving averages and slice contexts
df_raw = fetch_recent_15m(10)
df_feat = compute_live_features(df_raw, scaler_path)
feature_cols = get_feature_cols()

print("=========================================================================")
print("Comparing Old Settings vs Optimized Settings on Recent Data (LONG MODEL)")
print("Old: TP 1.50% / SL 0.75% / MaxBars 16")
print("New: TP 1.25% / SL 2.50% / MaxBars 12")
print("Threshold: > 0.60 Confidence")
print("=========================================================================\n")

for days in [1, 3, 7]:
    candles = days * 96
    
    start_idx = - (candles + seq_len)
    df_slice = df_feat.iloc[start_idx:]
    
    available = [c for c in feature_cols if c in df_slice.columns]
    X = df_slice[available].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    close = df_feat['close'].values[start_idx:]
    high = df_feat['high'].values[start_idx:]
    low = df_feat['low'].values[start_idx:]

    all_seqs = np.array([X[i:i+seq_len] for i in range(len(X) - seq_len)])
    seq_tensor = torch.tensor(all_seqs, dtype=torch.float32).to(device)

    signals = []
    with torch.no_grad():
        for i in range(0, len(seq_tensor), 2048):
            batch = seq_tensor[i:i+2048]
            out = model(batch)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().tolist()
            signals.extend(probs)

    res_old = backtest_tpsl(signals, close, high, low, seq_len, tp=0.015, sl=0.0075, max_bars=16)
    res_new = backtest_tpsl(signals, close, high, low, seq_len, tp=0.0125, sl=0.025, max_bars=12)
    
    print(f"--- LAST {days} DAY{'S' if days>1 else ''} ---")
    print(f"OLD: ROI {res_old['roi']:>5.2f}% | Win Rate {res_old['wins']}/{res_old['losses']} | Trades {res_old['trades']}")
    print(f"NEW: ROI {res_new['roi']:>5.2f}% | Win Rate {res_new['wins']}/{res_new['losses']} | Trades {res_new['trades']}")
    print("")
