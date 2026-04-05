"""Pre-compute AI bot signals and save as static JSON for the web dashboard."""
import json
import os
import sys
import time
import pandas as pd
import numpy as np
import requests as http_requests

sys.path.insert(0, '.')

def generate_signals():
    import torch
    from data.feature_engineer_btc import compute_live_features, get_feature_cols
    from ml.model import AttentionLSTMModel

    config_long_path = 'models/holy_grail_config.json'
    model_long_path = 'models/holy_grail.pth'
    config_short_path = 'models_short/holy_grail_short_config.json'
    model_short_path = 'models_short/holy_grail_short.pth'
    scaler_path = 'data_storage/BTC_USDT_15m_scaler.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(config_long_path, 'r') as f: config_long = json.load(f)
    with open(config_short_path, 'r') as f: config_short = json.load(f)
    s_long = config_long.get('seq_len', 128)
    s_short = config_short.get('seq_len', 128)

    model_long = AttentionLSTMModel(
        input_dim=config_long['input_dim'], hidden_dim=config_long['hidden_dim'],
        num_layers=config_long['num_layers'], output_dim=2,
        dropout=config_long['dropout'], num_heads=config_long['num_heads']
    ).to(device)
    model_long.load_state_dict(torch.load(model_long_path, map_location=device, weights_only=True))
    model_long.eval()

    model_short = AttentionLSTMModel(
        input_dim=config_short['input_dim'], hidden_dim=config_short['hidden_dim'],
        num_layers=config_short['num_layers'], output_dim=2,
        dropout=config_short['dropout'], num_heads=config_short['num_heads']
    ).to(device)
    model_short.load_state_dict(torch.load(model_short_path, map_location=device, weights_only=True))
    model_short.eval()
    
    print(f"Models loaded on {device}")

    # Fetch 20k candles of 15m data (backwards from now)
    all_raw = []
    interval = '15m'
    ims = 900000
    latest_url = f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit=1"
    try:
        latest = http_requests.get(latest_url, timeout=10).json()
        if latest and isinstance(latest, list):
            end_ts = int(latest[0][0])
            for page in range(20):
                page_end = end_ts - (page * 1000 * ims)
                page_start = page_end - (1000 * ims)
                url = f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit=1000&startTime={page_start}&endTime={page_end}"
                res = http_requests.get(url, timeout=15)
                data = res.json()
                if not data or not isinstance(data, list):
                    break
                all_raw = data + all_raw
                print(f"  Page {page+1}: {len(data)} candles (total: {len(all_raw)})")
                time.sleep(0.2)
    except Exception as e:
        print(f"Failed to fetch binance api history: {e}")

    # Deduplicate
    seen = set()
    unique_raw = []
    for r in all_raw:
        if r[0] not in seen:
            seen.add(r[0])
            unique_raw.append(r)
    unique_raw.sort(key=lambda x: x[0])
    print(f"Total unique candles: {len(unique_raw)}")
    
    if len(unique_raw) == 0:
        return

    df = pd.DataFrame(unique_raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                     'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    feat_df = compute_live_features(df, scaler_path)
    feature_cols = get_feature_cols()

    MAX_SEQ_LEN = max(s_long, s_short)
    signals = []

    if len(feat_df) >= MAX_SEQ_LEN:
        feat_unix = (feat_df.index.astype(np.int64) // 10**9).tolist()
        features_np = feat_df[feature_cols].values.astype(np.float32)

        batch_size = 64
        all_bull = []
        all_bear = []
        total_batches = (len(features_np) - MAX_SEQ_LEN + 1 + batch_size - 1) // batch_size
        for i, start in enumerate(range(0, len(features_np) - MAX_SEQ_LEN + 1, batch_size)):
            end = min(start + batch_size, len(features_np) - MAX_SEQ_LEN + 1)
            batch_l = [features_np[j + MAX_SEQ_LEN - s_long : j + MAX_SEQ_LEN] for j in range(start, end)]
            batch_s = [features_np[j + MAX_SEQ_LEN - s_short : j + MAX_SEQ_LEN] for j in range(start, end)]
            batch_tensor_l = torch.tensor(np.array(batch_l)).to(device)
            batch_tensor_s = torch.tensor(np.array(batch_s)).to(device)
            with torch.no_grad():
                logits_long = model_long(batch_tensor_l)
                bull_probs = torch.softmax(logits_long, dim=1)[:, 1].cpu().numpy().tolist()
                
                logits_short = model_short(batch_tensor_s)
                bear_probs = torch.softmax(logits_short, dim=1)[:, 1].cpu().numpy().tolist()
                
                all_bull.extend(bull_probs)
                all_bear.extend(bear_probs)
            if (i+1) % 50 == 0:
                print(f"  Inference: {i+1}/{total_batches} batches")

        print(f"Total predictions: {len(all_bull)}")

        import datetime
        position = 'flat'
        entry_price = 0
        bars_held = 0
        
        LONG_TP = 0.0125
        LONG_SL = 0.025
        LONG_MAX_BARS = 12
        SHORT_TP = 0.015
        SHORT_SL = 0.008
        SHORT_MAX_BARS = 8
        
        for idx in range(len(all_bull)):
            prob_bull = all_bull[idx]
            prob_bear = all_bear[idx]
            bar_idx = MAX_SEQ_LEN - 1 + idx
            
            if bar_idx < len(feat_unix):
                time_val = feat_unix[bar_idx]
                close_price = feat_df.iloc[bar_idx]['close'] if 'close' in feat_df.columns and bar_idx < len(feat_df) else 0
                high_price = feat_df.iloc[bar_idx]['high'] if 'high' in feat_df.columns and bar_idx < len(feat_df) else 0
                low_price = feat_df.iloc[bar_idx]['low'] if 'low' in feat_df.columns and bar_idx < len(feat_df) else 0
                
                utc_hour = datetime.datetime.utcfromtimestamp(time_val).hour
                can_trade = not (13 <= utc_hour <= 16)

                if position == 'flat':
                    if prob_bull >= 0.60 and prob_bear >= 0.50 and can_trade:
                        pass # Conflict, stay flat
                    elif prob_bull >= 0.60 and can_trade:
                        signals.append({'time': time_val, 'prob': round(prob_bull*100,1), 'signal': 'BUY', 'price': round(close_price, 2)})
                        position = 'long'
                        entry_price = close_price
                        bars_held = 0
                    elif prob_bear >= 0.50 and can_trade:
                        signals.append({'time': time_val, 'prob': round(prob_bear*100,1), 'signal': 'SELL', 'price': round(close_price, 2)})
                        position = 'short'
                        entry_price = close_price
                        bars_held = 0
                else:
                    bars_held += 1
                    closed = False
                    pnl = 0
                    
                    if position == 'long':
                        tp_price = entry_price * (1 + LONG_TP)
                        sl_price = entry_price * (1 - LONG_SL)
                        
                        if low_price <= sl_price:
                            pnl = round((sl_price - entry_price) / entry_price * 100, 2)
                            closed = True
                        elif high_price >= tp_price:
                            pnl = round((tp_price - entry_price) / entry_price * 100, 2)
                            closed = True
                        elif bars_held >= LONG_MAX_BARS:
                            pnl = round((close_price - entry_price) / entry_price * 100, 2)
                            closed = True
                    elif position == 'short':
                        tp_price = entry_price * (1 - SHORT_TP)
                        sl_price = entry_price * (1 + SHORT_SL)
                        
                        if high_price >= sl_price:
                            pnl = round((entry_price - sl_price) / entry_price * 100, 2)
                            closed = True
                        elif low_price <= tp_price:
                            pnl = round((entry_price - tp_price) / entry_price * 100, 2)
                            closed = True
                        elif bars_held >= SHORT_MAX_BARS:
                            pnl = round((entry_price - close_price) / entry_price * 100, 2)
                            closed = True
                            
                    if closed:
                        signals.append({'time': time_val, 'prob': 0, 'signal': 'CLOSE', 'price': round(close_price, 2), 'pnl': pnl})
                        position = 'flat'
                        entry_price = 0

    buys = len([s for s in signals if s['signal'] == 'BUY'])
    sells = len([s for s in signals if s['signal'] == 'SELL'])
    closes = len([s for s in signals if s['signal'] == 'CLOSE'])
    print(f"Generated {len(signals)} signals: {buys} BUY, {sells} SHORT, {closes} CLOSE")

    # Save
    out = {'signals': signals, 'generated_at': time.time(), 'candles': len(unique_raw)}
    os.makedirs('data_storage', exist_ok=True)
    with open('data_storage/precomputed_signals.json', 'w') as f:
        json.dump(out, f)
    print(f"Saved to data_storage/precomputed_signals.json ({len(json.dumps(out))//1024}KB)")

if __name__ == '__main__':
    generate_signals()
