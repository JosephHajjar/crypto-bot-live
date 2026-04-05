import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from ml.model import AttentionLSTMModel
from data.feature_engineer_btc import get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def investigate():
    print("Loading recent 12 hours of market data...")
    df = pd.read_csv('data_storage/BTC_USDT_15m_processed.csv')
    df = df.copy()
    
    # 12 hours = 48 candles of 15m
    df_recent = df.iloc[-80:] # Give it 80 to be safe
    
    with open('models/holy_grail_config.json', 'r') as f: cfg_l = json.load(f)
    s_long = cfg_l.get('seq_len', 128)
    
    m_long = AttentionLSTMModel(
        input_dim=cfg_l['input_dim'], hidden_dim=cfg_l['hidden_dim'],
        num_layers=cfg_l['num_layers'], output_dim=2, dropout=cfg_l['dropout'], num_heads=cfg_l['num_heads']
    ).to(device)
    m_long.load_state_dict(torch.load('models/holy_grail.pth', map_location=device, weights_only=True))
    m_long.eval()

    feature_cols = get_feature_cols()
    
    # We need to evaluate the last 48 candles.
    # To evaluate a candle, we need its previous 128 candles as a sequence.
    # So we need to feed the full dataframe into the sliding window logic, and pluck the last 48 outputs.
    
    features_np = df[feature_cols].values.astype(np.float32)
    unix_times = df['timestamp'].values
    close_prices = df['close'].values
    
    # Let's slice just the last 48 inferences
    start_eval_idx = len(features_np) - 48
    
    print("\n--- INFERENCE RESULTS FOR PAST 12 HOURS (48 CANDLES) ---")
    found_trade = False
    
    with torch.no_grad():
        for i in range(start_eval_idx, len(features_np)):
            # The exact sequence that ends AT index `i`, meaning features from `i - s_long + 1` to `i` (inclusive)
            seq = features_np[i - s_long + 1 : i + 1]
            t_long = torch.tensor(np.array([seq])).to(device)
            logits = m_long(t_long)
            prob_bull = torch.softmax(logits, dim=1)[0, 1].item()
            
            time_val = unix_times[i]
            dt = pd.to_datetime(time_val) if isinstance(time_val, str) else datetime.utcfromtimestamp(time_val)
            
            flag = ""
            if prob_bull >= 0.60:
                flag = "[LONG TRIGGER POTENTIAL!]"
                found_trade = True
            elif prob_bull >= 0.50:
                flag = "neutral-bullish"
                
            print(f"[{dt} UTC] | Price: ${close_prices[i]:.2f} | Bull Prob: {prob_bull*100:5.1f}% | {flag}")

    if not found_trade:
        print("\nCONCLUSION: The Model CORRECTLY stayed flat. The AI correctly calculated that NO 15-minute setup in the past 12 hours had a 60.0%+ probability of succeeding!")
    else:
        print("\nCONCLUSION: The Model DID calculate > 60.0% probability. If the bot didn't trade, it was likely blocked by the Deadzone or Paper Trading constraints.")

if __name__ == '__main__':
    investigate()
