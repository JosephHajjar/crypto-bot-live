from flask import Flask, jsonify, render_template, request
import json
import os
import requests as http_requests
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.volume import VolumeWeightedAveragePrice

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ---- Lazy-loaded AI Model for Bot Signals ----
_bot_model = None
_bot_config = None
_bot_scaler = None
_bot_device = None

def _load_bot_model():
    """Lazily load the trained AI model, config, and scaler. Cached after first call."""
    global _bot_model, _bot_config, _bot_scaler, _bot_device
    if _bot_model is not None:
        return True
    try:
        import torch
        import sys
        sys.path.insert(0, '.')
        from ml.model import AttentionLSTMModel
        
        config_path = 'models/trial_255_config.json'
        model_path = 'models/trial_255.pth'
        scaler_path = 'data_storage/BTC_USDT_15m_scaler.json'
        
        if not all(os.path.exists(p) for p in [config_path, model_path, scaler_path]):
            print("Bot model files not found")
            return False
        
        with open(config_path, 'r') as f:
            _bot_config = json.load(f)
        with open(scaler_path, 'r') as f:
            _bot_scaler = json.load(f)
            
        _bot_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _bot_model = AttentionLSTMModel(
            input_dim=_bot_config['input_dim'],
            hidden_dim=_bot_config['hidden_dim'],
            num_layers=_bot_config['num_layers'],
            output_dim=2,
            dropout=_bot_config['dropout'],
            num_heads=_bot_config['num_heads']
        ).to(_bot_device)
        _bot_model.load_state_dict(torch.load(model_path, map_location=_bot_device, weights_only=True))
        _bot_model.eval()
        print(f"Bot model loaded successfully on {_bot_device}")
        return True
    except Exception as e:
        print(f"Failed to load bot model: {e}")
        import traceback; traceback.print_exc()
        return False

# ---- Supertrend ----
def compute_supertrend(df, length=10, multiplier=3.0):
    """Supertrend with directional coloring data."""
    hl2 = (df['high'] + df['low']) / 2
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(length).mean()
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)
    
    for i in range(length, len(df)):
        if i == length:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
            continue
        if direction.iloc[i-1] == -1:
            if df['close'].iloc[i] > supertrend.iloc[i-1]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
        else:
            if df['close'].iloc[i] < supertrend.iloc[i-1]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
    return supertrend, direction

# ---- Routes ----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    if os.path.exists('data_storage/live_state.json'):
        try:
            with open('data_storage/live_state.json', 'r') as f:
                return jsonify(json.load(f))
        except Exception:
            return jsonify({"error": "State file locked or corrupt"})
    return jsonify({"error": "No state active"})

@app.route('/api/historical_data')
def get_historical():
    symbol = request.args.get('symbol', 'BTCUSDT')
    interval = request.args.get('interval', '15m')
    limit = int(request.args.get('limit', 1000))
    start_time = request.args.get('startTime', '')
    
    # Paginated fetch: get up to 5000 candles via multiple requests
    all_data = []
    pages = min((limit + 999) // 1000, 5)  # max 5 pages = 5000 candles
    per_page = min(limit, 1000)
    current_start = start_time
    
    try:
        for page in range(pages):
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={per_page}"
            if current_start:
                url += f"&startTime={current_start}"
            elif page == 0 and not start_time:
                # No start time: fetch backwards by computing start from limit
                # First request gets latest 1000, we'll paginate backwards
                pass
            
            res = http_requests.get(url, timeout=15)
            data = res.json()
            if not data or not isinstance(data, list):
                break
            all_data.extend(data)
            if len(data) < per_page:
                break  # No more data available
            # Next page starts after the last candle
            current_start = str(int(data[-1][0]) + 1)
        
        if not start_time and pages > 1:
            # Fetch backwards to get older data
            all_data = []
            # Start by getting the latest candle timestamp
            latest_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1"
            latest_res = http_requests.get(latest_url, timeout=10)
            latest = latest_res.json()
            if latest:
                # Calculate interval in ms
                interval_ms = {'1m':60000,'3m':180000,'5m':300000,'15m':900000,'30m':1800000,
                               '1h':3600000,'2h':7200000,'4h':14400000,'6h':21600000,
                               '8h':28800000,'12h':43200000,'1d':86400000,'3d':259200000,'1w':604800000}
                ims = interval_ms.get(interval, 900000)
                end_time = int(latest[0][0])
                for page in range(pages):
                    page_end = end_time - (page * 1000 * ims)
                    page_start = page_end - (1000 * ims)
                    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1000&startTime={page_start}&endTime={page_end}"
                    res = http_requests.get(url, timeout=15)
                    data = res.json()
                    if data and isinstance(data, list):
                        all_data = data + all_data
                    if not data or len(data) < 10:
                        break
        elif start_time:
            pass  # already paginated forward from startTime
        else:
            pass  # single page was fetched in all_data
        
        # Deduplicate by timestamp
        seen = set()
        unique_data = []
        for row in all_data:
            ts = row[0]
            if ts not in seen:
                seen.add(ts)
                unique_data.append(row)
        unique_data.sort(key=lambda x: x[0])
        
        df = pd.DataFrame(unique_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                          'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore'])
        df['timestamp'] = df['timestamp'].astype(float) / 1000
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # EMA 9 & 21
        df['ema9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
        
        # VWAP
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        try:
            vwap_ind = VolumeWeightedAveragePrice(high=df['high'], low=df['low'],
                                                   close=df['close'], volume=df['volume'])
            df['vwap'] = vwap_ind.volume_weighted_average_price()
        except Exception:
            df['vwap'] = 0
        
        # MACD
        macd_ind = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()
        df['macd_hist'] = macd_ind.macd_diff()
        
        # Supertrend with direction
        st_val, st_dir = compute_supertrend(df, length=10, multiplier=3.0)
        df['supertrend'] = st_val
        df['supertrend_dir'] = st_dir
            
        df = df.fillna(0)
        records = df.to_dict(orient='records')
        return jsonify(records)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/bot_signals')
def get_bot_signals():
    """Run the trained AI model on historical 15m data and return buy/sell signals."""
    if not _load_bot_model():
        return jsonify({"error": "Model not available", "signals": []})
    
    try:
        import torch
        from data.feature_engineer import compute_live_features, get_feature_cols
        
        start_time = request.args.get('startTime', '')
        
        # Fetch more data for better feature coverage (3000 candles)
        all_raw = []
        if start_time:
            # Paginate forward from start time
            current = start_time
            for _ in range(5):
                url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=1000&startTime={current}"
                res = http_requests.get(url, timeout=15)
                data = res.json()
                if not data: break
                all_raw.extend(data)
                if len(data) < 1000: break
                current = str(int(data[-1][0]) + 1)
        else:
            # Fetch latest 3000 candles backwards
            latest_url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=1"
            latest = http_requests.get(latest_url, timeout=10).json()
            if latest:
                end_ts = int(latest[0][0])
                for page in range(5):
                    page_end = end_ts - (page * 1000 * 900000)
                    page_start = page_end - (1000 * 900000)
                    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=1000&startTime={page_start}&endTime={page_end}"
                    res = http_requests.get(url, timeout=15)
                    data = res.json()
                    if data and isinstance(data, list):
                        all_raw = data + all_raw
        
        # Deduplicate
        seen = set()
        unique_raw = []
        for r in all_raw:
            if r[0] not in seen:
                seen.add(r[0])
                unique_raw.append(r)
        unique_raw.sort(key=lambda x: x[0])
        
        df = pd.DataFrame(unique_raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                         'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        unix_times = (df['timestamp'].astype(np.int64) // 10**9).tolist()
        
        # Compute features
        feat_df = compute_live_features(df, 'data_storage/BTC_USDT_15m_scaler.json')
        feature_cols = get_feature_cols()
        
        SEQ_LEN = 128
        signals = []
        
        if len(feat_df) >= SEQ_LEN:
            feat_unix = (feat_df.index.astype(np.int64) // 10**9).tolist()
            features_np = feat_df[feature_cols].values.astype(np.float32)
            
            # Batch inference
            batch_size = 64
            all_probs = []
            for start in range(0, len(features_np) - SEQ_LEN + 1, batch_size):
                end = min(start + batch_size, len(features_np) - SEQ_LEN + 1)
                batch = [features_np[i:i+SEQ_LEN] for i in range(start, end)]
                batch_tensor = torch.tensor(np.array(batch)).to(_bot_device)
                
                with torch.no_grad():
                    logits = _bot_model(batch_tensor)
                    probs = torch.softmax(logits, dim=1)
                    bull_probs = probs[:, 1].cpu().numpy().tolist()
                    all_probs.extend(bull_probs)
            
            # --- Simulate full trade lifecycle ---
            # Track position state to generate realistic BUY/SELL/CLOSE signals
            position = 'flat'  # flat, long, short
            entry_price = 0
            for idx, prob in enumerate(all_probs):
                bar_idx = SEQ_LEN - 1 + idx
                if bar_idx < len(feat_unix):
                    time_val = feat_unix[bar_idx]
                    # Get the close price at this bar for P&L tracking
                    close_price = feat_df.iloc[bar_idx]['close'] if 'close' in feat_df.columns and bar_idx < len(feat_df) else 0
                    
                    import datetime
                    utc_hour = datetime.datetime.utcfromtimestamp(time_val).hour
                    can_trade = not (13 <= utc_hour <= 16)

                    if position == 'flat':
                        if prob > 0.55 and can_trade:
                            signals.append({'time': time_val, 'prob': round(prob*100,1), 'signal': 'BUY', 'price': round(close_price, 2)})
                            position = 'long'
                            entry_price = close_price
                        elif prob < 0.35 and can_trade:
                            signals.append({'time': time_val, 'prob': round(prob*100,1), 'signal': 'SELL', 'price': round(close_price, 2)})
                            position = 'short'
                            entry_price = close_price
                    elif position == 'long':
                        if prob < 0.40:
                            pnl = round((close_price - entry_price) / entry_price * 100, 2) if entry_price > 0 else 0
                            signals.append({'time': time_val, 'prob': round(prob*100,1), 'signal': 'CLOSE', 'price': round(close_price, 2), 'pnl': pnl})
                            position = 'flat'
                            entry_price = 0
                    elif position == 'short':
                        if prob > 0.60:
                            pnl = round((entry_price - close_price) / entry_price * 100, 2) if entry_price > 0 else 0
                            signals.append({'time': time_val, 'prob': round(prob*100,1), 'signal': 'CLOSE', 'price': round(close_price, 2), 'pnl': pnl})
                            position = 'flat'
                            entry_price = 0
        
        return jsonify({"signals": signals})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "signals": []})

if __name__ == '__main__':
    print("Dashboard Running! Go to: http://127.0.0.1:5001")
    app.run(port=5001, debug=False, use_reloader=False, threaded=True)
