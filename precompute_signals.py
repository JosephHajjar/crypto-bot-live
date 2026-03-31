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
    from data.feature_engineer import compute_live_features, get_feature_cols
    from ml.model import AttentionLSTMModel

    config_path = 'models/trial_255_config.json'
    model_path = 'models/trial_255.pth'
    scaler_path = 'data_storage/BTC_USDT_15m_scaler.json'

    with open(config_path, 'r') as f:
        config = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionLSTMModel(
        input_dim=config['input_dim'], hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'], output_dim=2,
        dropout=config['dropout'], num_heads=config['num_heads']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded on {device}")

    # Fetch 20k candles of 15m data (backwards from now)
    all_raw = []
    interval = '15m'
    ims = 900000
    latest_url = f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit=1"
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

    # Deduplicate
    seen = set()
    unique_raw = []
    for r in all_raw:
        if r[0] not in seen:
            seen.add(r[0])
            unique_raw.append(r)
    unique_raw.sort(key=lambda x: x[0])
    print(f"Total unique candles: {len(unique_raw)}")

    df = pd.DataFrame(unique_raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                     'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    feat_df = compute_live_features(df, scaler_path)
    feature_cols = get_feature_cols()

    SEQ_LEN = 128
    signals = []

    if len(feat_df) >= SEQ_LEN:
        feat_unix = (feat_df.index.astype(np.int64) // 10**9).tolist()
        features_np = feat_df[feature_cols].values.astype(np.float32)

        batch_size = 64
        all_probs = []
        total_batches = (len(features_np) - SEQ_LEN + 1 + batch_size - 1) // batch_size
        for i, start in enumerate(range(0, len(features_np) - SEQ_LEN + 1, batch_size)):
            end = min(start + batch_size, len(features_np) - SEQ_LEN + 1)
            batch = [features_np[j:j+SEQ_LEN] for j in range(start, end)]
            batch_tensor = torch.tensor(np.array(batch)).to(device)
            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=1)
                bull_probs = probs[:, 1].cpu().numpy().tolist()
                all_probs.extend(bull_probs)
            if (i+1) % 50 == 0:
                print(f"  Inference: {i+1}/{total_batches} batches")

        print(f"Total predictions: {len(all_probs)}")

        import datetime
        position = 'flat'
        entry_price = 0
        bars_held = 0
        for idx, prob in enumerate(all_probs):
            bar_idx = SEQ_LEN - 1 + idx
            if bar_idx < len(feat_unix):
                time_val = feat_unix[bar_idx]
                close_price = feat_df.iloc[bar_idx]['close'] if 'close' in feat_df.columns and bar_idx < len(feat_df) else 0
                utc_hour = datetime.datetime.utcfromtimestamp(time_val).hour
                can_trade = not (13 <= utc_hour <= 16)

                if position == 'flat':
                    if prob > 0.50 and can_trade:
                        signals.append({'time': time_val, 'prob': round(prob*100,1), 'signal': 'BUY', 'price': round(close_price, 2)})
                        position = 'long'
                        entry_price = close_price
                        bars_held = 0
                    elif prob < 0.45 and can_trade:
                        signals.append({'time': time_val, 'prob': round(prob*100,1), 'signal': 'SELL', 'price': round(close_price, 2)})
                        position = 'short'
                        entry_price = close_price
                        bars_held = 0
                else:
                    bars_held += 1
                    if position == 'long':
                        if prob < 0.50 or bars_held >= 16:
                            pnl = round((close_price - entry_price) / entry_price * 100, 2) if entry_price > 0 else 0
                            signals.append({'time': time_val, 'prob': round(prob*100,1), 'signal': 'CLOSE', 'price': round(close_price, 2), 'pnl': pnl})
                            position = 'flat'
                            entry_price = 0
                    elif position == 'short':
                        if prob > 0.50 or bars_held >= 16:
                            pnl = round((entry_price - close_price) / entry_price * 100, 2) if entry_price > 0 else 0
                            signals.append({'time': time_val, 'prob': round(prob*100,1), 'signal': 'CLOSE', 'price': round(close_price, 2), 'pnl': pnl})
                            position = 'flat'
                            entry_price = 0

    buys = len([s for s in signals if s['signal'] == 'BUY'])
    sells = len([s for s in signals if s['signal'] == 'SELL'])
    closes = len([s for s in signals if s['signal'] == 'CLOSE'])
    print(f"Generated {len(signals)} signals: {buys} BUY, {sells} SELL, {closes} CLOSE")

    # Save
    out = {'signals': signals, 'generated_at': time.time(), 'candles': len(unique_raw)}
    os.makedirs('data_storage', exist_ok=True)
    with open('data_storage/precomputed_signals.json', 'w') as f:
        json.dump(out, f)
    print(f"Saved to data_storage/precomputed_signals.json ({len(json.dumps(out))//1024}KB)")

if __name__ == '__main__':
    generate_signals()
