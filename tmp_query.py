import sys
sys.path.insert(0, '.')
from trade_live import LiveHyperliquidTrader
import pandas as pd
import numpy as np
import torch
from data.feature_engineer_btc import compute_live_features, get_feature_cols

trader = LiveHyperliquidTrader()
df = trader.fetch_recent_data()
live_df = compute_live_features(df, 'data_storage/BTC_USDT_15m_scaler.json')
feature_cols = get_feature_cols()
feat_np = live_df[feature_cols].values.astype(np.float32)

print("\n--- PAST 2 HOURS (8 CANDLES) ---")
for i in range(len(feat_np)-8, len(feat_np)):
    feat_long = feat_np[i-trader.seq_len_long+1:i+1]
    feat_short = feat_np[i-trader.seq_len_short+1:i+1]
    
    t_long = torch.tensor(feat_long).unsqueeze(0).to(trader.device)
    t_short = torch.tensor(feat_short).unsqueeze(0).to(trader.device)
    
    bull = torch.softmax(trader.model_long(t_long), dim=1)[0][1].item()
    bear = torch.softmax(trader.model_short(t_short), dim=1)[0][1].item()
    
    time_str = df['timestamp'].iloc[i]
    close = df['close'].iloc[i]
    
    flag = ""
    if bull >= 0.60 and bear >= 0.50:
        flag += " [CONFLICT] "
    elif bull >= 0.60:
        flag += " [LONG ENTRY] "
    elif bear >= 0.50:
        flag += " [SHORT ENTRY] "
    else:
        flag += " [FLAT] "
        
    print(f"[{time_str}] Price: ${close:.2f} | Bull: {bull*100:.1f}%, Bear: {bear*100:.1f}% {flag}")
