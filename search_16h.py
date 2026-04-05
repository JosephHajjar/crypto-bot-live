import sys
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
from ml.model import AttentionLSTMModel
from data.feature_engineer import compute_live_features, get_feature_cols
from trade_live import LiveHyperliquidTrader, SYMBOL, TIMEFRAME

try:
    trader = LiveHyperliquidTrader()
    print('Loaded Live Hyperliquid AI Trader on cuda. Active Trade:', trader.in_trade)
    
    # Check what the model thinks of recent data
    df = trader.fetch_recent_data()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    feat_df = compute_live_features(df, 'data_storage/BTC_USDT_15m_scaler.json')
    feature_cols = get_feature_cols()

    seq_long = trader.seq_len_long
    seq_short = trader.seq_len_short
    max_seq = max(seq_long, seq_short)

    found_signals = False
    for i in range(1, 65):
        if len(feat_df) < max_seq + i:
            continue
        
        test_df = feat_df.iloc[:len(feat_df) - i]
        cur_bar = test_df.iloc[-1]
        
        feats_l = test_df[feature_cols].values[-seq_long:]
        feats_s = test_df[feature_cols].values[-seq_short:]
        
        tens_l = torch.tensor(feats_l, dtype=torch.float32).unsqueeze(0).to(trader.device)
        tens_s = torch.tensor(feats_s, dtype=torch.float32).unsqueeze(0).to(trader.device)
        
        with torch.no_grad():
            out_l = trader.model_long(tens_l)
            bull_prob = torch.softmax(out_l, dim=1)[0][1].item()
            
            out_s = trader.model_short(tens_s)
            bear_prob = torch.softmax(out_s, dim=1)[0][1].item()
            
        action = '[FLAT]'
        if bull_prob >= 0.60 and bear_prob < 0.50:
            action = '[BUY LONG SIGNAL]'
            found_signals = True
        elif bear_prob >= 0.50 and bull_prob < 0.60:
            action = '[SELL SHORT SIGNAL]'
            found_signals = True
            
        if bull_prob >= 0.60 or bear_prob >= 0.50:
            print(f'[{cur_bar.name}] Price: \${cur_bar.close:.2f} | Bull: {bull_prob*100:.1f}%, Bear: {bear_prob*100:.1f}%  {action}')
        
    print('Result: Found signals:', found_signals)
except Exception as e:
    import traceback
    traceback.print_exc()
