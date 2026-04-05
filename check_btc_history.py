import torch
import pandas as pd
import numpy as np
import json
from ml.model import AttentionLSTMModel
from data.fetch_data import fetch_klines
import pytz
from data.fetch_data import fetch_klines
from data.feature_engineer_btc import engineer_features, get_feature_cols

def check_history():
    print("Loading recent BTC historical context...")
    
    df_raw = fetch_klines('BTC/USDT', '15m', 2, 'data_storage')
    df = pd.read_csv(engineer_features(df_raw)[0])
    
    with open('models/holy_grail_config.json', 'r') as f:
        cfg_l = json.load(f)
    s_long = cfg_l.get('seq_len', 128)
    
    m_long = AttentionLSTMModel(
        input_dim=cfg_l['input_dim'], hidden_dim=cfg_l['hidden_dim'],
        num_layers=cfg_l['num_layers'], output_dim=2, dropout=cfg_l['dropout'], num_heads=cfg_l['num_heads']
    ).to('cpu')
    m_long.load_state_dict(torch.load('models/holy_grail.pth', map_location='cpu', weights_only=True))
    m_long.eval()

    with open('models_short/holy_grail_short_config.json', 'r') as f:
        cfg_s = json.load(f)
    s_short = cfg_s.get('seq_len', 128)
    
    m_short = AttentionLSTMModel(
        input_dim=cfg_s['input_dim'], hidden_dim=cfg_s['hidden_dim'],
        num_layers=cfg_s['num_layers'], output_dim=2, dropout=cfg_s['dropout'], num_heads=cfg_s['num_heads']
    ).to('cpu')
    m_short.load_state_dict(torch.load('models_short/holy_grail_short.pth', map_location='cpu', weights_only=True))
    m_short.eval()

    cols = get_feature_cols()
    feats = df[cols].values.astype(np.float32)
    
    dubai_tz = pytz.timezone("Asia/Dubai")
    
    results = []
    
    with torch.no_grad():
        for i in range(len(df)-10, len(df)):
            if i < max(s_long, s_short): continue
            
            seq_l = feats[i - s_long:i]
            seq_s = feats[i - s_short:i]
            
            p_bull = torch.softmax(m_long(torch.tensor(np.array([seq_l]))), dim=1)[0][1].item() * 100
            p_bear = torch.softmax(m_short(torch.tensor(np.array([seq_s]))), dim=1)[0][1].item() * 100
            
            ts = df.iloc[i]['timestamp']
            ts_dt = pd.to_datetime(ts)
            if ts_dt.tzinfo is None:
                dt = ts_dt.tz_localize('UTC').tz_convert(dubai_tz)
            else:
                dt = ts_dt.tz_convert(dubai_tz)
            
            time_str = dt.strftime("%I:%M %p")
            results.append(f"Time: {time_str} | LONG EDGE: {p_bull:.3f}% | SHORT EDGE: {p_bear:.3f}%")
            
    print("\n========= HISTORICAL AI EDGE (BTC) - DUBAI TIME =========")
    # They requested from 4:30 pm backward to 3:30 pm. 
    # Let's print exactly matching those if they exist in the 15m intervals
    for res in reversed(results):
        print(res)

if __name__ == '__main__':
    check_history()
