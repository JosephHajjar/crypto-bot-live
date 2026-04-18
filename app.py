from flask import Flask, jsonify, render_template, request
import json
import os
import requests as http_requests
import pandas as pd
import numpy as np
import time
from ta.trend import EMAIndicator, MACD
from ta.volume import VolumeWeightedAveragePrice
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ---- Cached Hyperliquid state (refreshes every 10s, not every request) ----
_hl_cache = {"data": None, "last_fetch": 0}
_HL_CACHE_TTL = 10  # seconds
# Track when we first saw the current position so we can compute bars_held
_position_tracker = {"entry_time": None, "entry_price": None, "direction": None}

def _get_exchange_state():
    """Fetch real Hyperliquid account state with 10s caching."""
    now = time.time()
    if _hl_cache["data"] is not None and (now - _hl_cache["last_fetch"]) < _HL_CACHE_TTL:
        return _hl_cache["data"]
    
    try:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        wallet = os.environ.get("HYPERLIQUID_WALLET_ADDRESS", "").strip()
        if not wallet:
            return None
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        user_state = info.user_state(wallet)
        margin = user_state.get("marginSummary", {})
        # Perps balance
        perps_value = float(margin.get("accountValue", 0.0))
        
        # Spot USDC balance
        spot_usdc = 0.0
        try:
            spot_state = info.spot_user_state(wallet)
            for bal in spot_state.get("balances", []):
                if bal.get("coin") == "USDC":
                    spot_usdc = float(bal.get("total", 0.0))
                    break
        except Exception:
            pass
        
        # Total = perps + spot
        account_value = perps_value + spot_usdc
        
        exchange_pos = None
        for pos in user_state.get("assetPositions", []):
            if pos['position']['coin'] == 'BTC':
                exchange_pos = pos['position']
                break
        
        exchange_size = float(exchange_pos['szi']) if exchange_pos else 0.0
        exchange_entry = float(exchange_pos['entryPx']) if exchange_pos else 0.0
        exchange_unrealized = float(exchange_pos.get('unrealizedPnl', 0)) if exchange_pos else 0.0
        has_position = abs(exchange_size) >= 0.00001
        
        # Track position entry time for bars_held calculation
        if has_position:
            direction = 'LONG' if exchange_size > 0 else 'SHORT'
            # If this is a new position (or direction/price changed significantly), record entry time
            if (_position_tracker["entry_time"] is None or 
                _position_tracker["direction"] != direction or
                (_position_tracker["entry_price"] is not None and 
                 abs(exchange_entry - _position_tracker["entry_price"]) / max(exchange_entry, 1) > 0.01)):
                _position_tracker["entry_time"] = now
                _position_tracker["entry_price"] = exchange_entry
                _position_tracker["direction"] = direction
        else:
            # Position closed, reset tracker
            _position_tracker["entry_time"] = None
            _position_tracker["entry_price"] = None
            _position_tracker["direction"] = None
        
        # Calculate bars_held: how many 15-minute bars since entry
        bars_held = 0
        if has_position and _position_tracker["entry_time"] is not None:
            elapsed_seconds = now - _position_tracker["entry_time"]
            bars_held = max(1, int(elapsed_seconds / 900))  # 900s = 15 minutes
        
        result = {
            "account_value": account_value,
            "has_position": has_position,
            "size": exchange_size,
            "entry": exchange_entry,
            "unrealized": exchange_unrealized,
            "bars_held": bars_held,
        }
        _hl_cache["data"] = result
        _hl_cache["last_fetch"] = now
        return result
    except Exception as e:
        print(f"Hyperliquid fetch error: {e}")
        return _hl_cache["data"]  # Return stale data on error

# ---- Lazy-loaded AI Models for Bot Signals ----
_bot_model_long = None
_bot_model_short = None
_bot_device = None

def _load_bot_model():
    """Lazily load trained AI models. Cached after first call."""
    global _bot_model_long, _bot_model_short, _bot_device
    if _bot_model_long is not None and _bot_model_short is not None:
        return True
    try:
        import torch
        import sys
        sys.path.insert(0, '.')
        from ml.model import AttentionLSTMModel
        
        config_long_path = 'models/holy_grail_config.json'
        model_long_path = 'models/holy_grail.pth'
        config_short_path = 'models_short/holy_grail_short_config.json'
        model_short_path = 'models_short/holy_grail_short.pth'
        
        if not all(os.path.exists(p) for p in [config_long_path, model_long_path, config_short_path, model_short_path]):
            print("Bot model files not found")
            return False
            
        _bot_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with open(config_long_path, 'r') as f: cfg_long = json.load(f)
        with open(config_short_path, 'r') as f: cfg_short = json.load(f)
        global _seq_len_long, _seq_len_short
        _seq_len_long = cfg_long.get('seq_len', 128)
        _seq_len_short = cfg_short.get('seq_len', 128)

        _bot_model_long = AttentionLSTMModel(
            input_dim=cfg_long['input_dim'], hidden_dim=cfg_long['hidden_dim'],
            num_layers=cfg_long['num_layers'], output_dim=2,
            dropout=cfg_long['dropout'], num_heads=cfg_long['num_heads']
        ).to(_bot_device)
        _bot_model_long.load_state_dict(torch.load(model_long_path, map_location=_bot_device, weights_only=True))
        _bot_model_long.eval()

        _bot_model_short = AttentionLSTMModel(
            input_dim=cfg_short['input_dim'], hidden_dim=cfg_short['hidden_dim'],
            num_layers=cfg_short['num_layers'], output_dim=2,
            dropout=cfg_short['dropout'], num_heads=cfg_short['num_heads']
        ).to(_bot_device)
        _bot_model_short.load_state_dict(torch.load(model_short_path, map_location=_bot_device, weights_only=True))
        _bot_model_short.eval()

        print(f"Dual-Bot models loaded successfully on {_bot_device}")
        return True
    except Exception as e:
        print(f"Failed to load bot models: {e}")
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
    symbol = request.args.get('symbol', 'BTCUSDT')
    file_name = 'data_storage/live_state_alt.json'
    state = {}
    if os.path.exists(file_name):
        try:
            with open(file_name, 'r') as f:
                state = json.load(f)
        except Exception:
            return jsonify({"error": "State file locked or corrupt"})
    
    # If the bot hasn't written state yet, provide minimum viable state
    # so the dashboard doesn't show "Offline"
    if 'timestamp' not in state:
        from datetime import datetime, timezone
        state['timestamp'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        state.setdefault('bull_prob', 0.0)
        state.setdefault('bear_prob', 0.0)
        state.setdefault('master_control', 'ALT')
        state.setdefault('in_trade', False)
        state.setdefault('trade_type', None)
        state.setdefault('entry_price', 0.0)
        state.setdefault('bars_held', 0)
        state.setdefault('open_pnl_pct', 0.0)
        state.setdefault('open_pnl_usd', 0.0)
        state.setdefault('trade_amount_btc', 0.0)
        state.setdefault('trade_amount_usd', 0.0)
        state.setdefault('last_error', None)
        # Fetch live price from Binance so chart still works
        try:
            pr = http_requests.get('https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT', timeout=5)
            state['current_price'] = float(pr.json()['price'])
        except Exception:
            state.setdefault('current_price', 0.0)

    # Always overlay real Hyperliquid data so dashboard shows truth
    hl = _get_exchange_state()
    if hl is not None:
        state['paper_balance'] = hl['account_value']
        state['in_trade'] = hl['has_position']
        if hl['has_position']:
            direction = 'LONG' if hl['size'] > 0 else 'SHORT'
            state['trade_type'] = direction
            state['entry_price'] = hl['entry']
            state['trade_amount_btc'] = abs(hl['size'])
            state['trade_amount_usd'] = round(abs(hl['size']) * hl['entry'], 2)
            state['open_pnl_usd'] = round(hl['unrealized'], 4)
            
            # Compute open_pnl_pct from entry vs current price
            current_price = state.get('current_price', 0)
            if current_price > 0 and hl['entry'] > 0:
                if direction == 'LONG':
                    state['open_pnl_pct'] = round((current_price - hl['entry']) / hl['entry'] * 100, 3)
                else:
                    state['open_pnl_pct'] = round((hl['entry'] - current_price) / hl['entry'] * 100, 3)
            
            # Compute bars_held from trade log or position tracker
            bars_held = hl.get('bars_held', 0)
            # Also try to find the actual entry time from trade log
            try:
                trades_file = 'data_storage/live_trades_ensemble.json'
                if os.path.exists(trades_file):
                    with open(trades_file, 'r') as f:
                        trades = json.load(f)
                    if trades:
                        last_trade = trades[-1]
                        last_ts_str = last_trade.get('timestamp', '')
                        if last_ts_str:
                            from datetime import datetime, timezone
                            last_ts = datetime.fromisoformat(last_ts_str.replace('Z', '+00:00'))
                            now_utc = datetime.now(timezone.utc)
                            elapsed = (now_utc - last_ts).total_seconds()
                            computed_bars = max(1, int(elapsed / 900))
                            bars_held = max(bars_held, computed_bars)
            except Exception:
                pass
            
            state['bars_held'] = bars_held
        else:
            state['trade_type'] = None
            state['entry_price'] = 0.0
            state['trade_amount_btc'] = 0.0
            state['trade_amount_usd'] = 0.0
            state['open_pnl_usd'] = 0.0
            state['open_pnl_pct'] = 0.0
            state['bars_held'] = 0
    
    if not state:
        return jsonify({"error": "No state active"})
    return jsonify(state)

@app.route('/api/klines')
def get_klines():
    symbol = request.args.get('symbol', 'BTCUSDT')
    limit = request.args.get('limit', '200')
    try:
        res = http_requests.get(f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=15m&limit={limit}", timeout=5)
        return jsonify(res.json())
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/debug')
def get_debug():
    """Read the last 100 lines of bot log for remote diagnostics."""
    log_file = 'alt_bot.log'
    lines = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()[-100:]
        except Exception as e:
            return jsonify({"error": str(e)})
    
    # Also check if the state file exists and its size
    state_exists = os.path.exists('data_storage/live_state_alt.json')
    state_size = os.path.getsize('data_storage/live_state_alt.json') if state_exists else 0
    
    return jsonify({
        "log_lines": [l.strip() for l in lines],
        "state_file_exists": state_exists,
        "state_file_size": state_size,
        "data_storage_contents": os.listdir('data_storage') if os.path.exists('data_storage') else [],
        "crash_log": open('alt_bot_crash.log', 'r').read() if os.path.exists('alt_bot_crash.log') else None,
        "env_wallet_set": bool(os.environ.get("HYPERLIQUID_WALLET_ADDRESS", "").strip()),
        "env_secret_set": bool(os.environ.get("HYPERLIQUID_API_SECRET", "").strip()),
        "cwd": os.getcwd(),
        "cwd_contents": os.listdir('.'),
    })

@app.route('/api/set_target', methods=['POST'])
def set_target():
    try:
        data = request.json
        target_price = float(data.get('target', 0))
        if target_price > 0:
            with open('data_storage/manual_override.json', 'w') as f:
                json.dump({"manual_target": target_price, "timestamp": time.time()}, f)
            return jsonify({"success": True, "target": target_price})
        elif target_price == 0:
            if os.path.exists('data_storage/manual_override.json'):
                os.remove('data_storage/manual_override.json')
            return jsonify({"success": True, "cleared": True})
    except Exception as e:
        return jsonify({"error": str(e)})
    return jsonify({"error": "Invalid request"})

@app.route('/api/historical_data')
def get_historical():
    symbol = request.args.get('symbol', 'BTCUSDT')
    interval = request.args.get('interval', '15m')
    limit = int(request.args.get('limit', 1000))
    start_time = request.args.get('startTime', '')
    
    symbol_fmt = "BTC_USDT" if symbol == "BTCUSDT" else symbol
    csv_path = f"data_storage/{symbol_fmt}_{interval}.csv"
    try:
        if os.path.exists(csv_path):
            import pandas as pd
            import numpy as np
            
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp']).values.astype(np.int64) // 10**6
            
            if start_time:
                df = df[df['timestamp'] >= float(start_time)]
                df = df.head(limit).copy()
            else:
                df = df.tail(limit).copy()
            
            df['timestamp'] = df['timestamp'].astype(float) / 1000
        else:
            return jsonify({"error": f"Local data not found for {symbol} {interval}"})
            
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df['ema9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        try:
            vwap_ind = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
            df['vwap'] = vwap_ind.volume_weighted_average_price()
        except Exception:
            df['vwap'] = 0
        
        macd_ind = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()
        df['macd_hist'] = macd_ind.macd_diff()
        
        st_val, st_dir = compute_supertrend(df, length=10, multiplier=3.0)
        df['supertrend'] = st_val
        df['supertrend_dir'] = st_dir
            
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        records = df.to_dict(orient='records')
        return jsonify(records)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/live_trades')
def get_live_trades():
    symbol = request.args.get('symbol', 'BTCUSDT')
    file_name = 'data_storage/live_trades_alt.json'
    if os.path.exists(file_name):
        try:
            with open(file_name, "r") as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)})
    return jsonify([])

@app.route('/api/bot_signals')
def get_bot_signals():
    """Run dual AI models on historical data and return LONG/SHORT signals."""
    if not _load_bot_model():
        precomputed_path = 'data_storage/precomputed_signals.json'
        if os.path.exists(precomputed_path):
            try:
                with open(precomputed_path, 'r') as f:
                    cached = json.load(f)
                start_time = request.args.get('startTime', '')
                if start_time:
                    start_ts = int(start_time) / 1000
                    cached['signals'] = [s for s in cached['signals'] if s['time'] >= start_ts]
                return jsonify(cached)
            except Exception:
                pass
        return jsonify({"error": "Model not available", "signals": []})
    
    try:
        import torch
        from data.feature_engineer_btc import compute_live_features, get_feature_cols
        
        start_time = request.args.get('startTime', '')
        interval = request.args.get('interval', '15m')
        limit = int(request.args.get('limit', 2000))
        
        symbol_fmt = "BTC_USDT"
        csv_path = f"data_storage/{symbol_fmt}_{interval}.csv"
        if not os.path.exists(csv_path):
            return jsonify({"error": "Local data not found", "signals": []})
            
        import pandas as pd
        import numpy as np
        
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        timestamp_ms = df['timestamp'].values.astype(np.int64) // 10**6
        if start_time:
            start_idx = np.searchsorted(timestamp_ms, float(start_time))
            safe_start = max(0, start_idx - 500)
            end_idx = min(len(df), start_idx + limit)
            df = df.iloc[safe_start:end_idx].copy()
        else:
            safe_start = max(0, len(df) - limit - 500)
            df = df.iloc[safe_start:].copy()
        
        unix_times = (df['timestamp'].astype(np.int64) // 10**9).tolist()
        
        feat_df = compute_live_features(df, 'data_storage/BTC_USDT_15m_scaler.json')
        feature_cols = get_feature_cols()
        
        import json
        with open('models/holy_grail_config.json', 'r') as f:
            cfg_l = json.load(f)
        with open('models_short/holy_grail_short_config.json', 'r') as f:
            cfg_s = json.load(f)
        s_long = cfg_l.get('seq_len', 128)
        s_short = cfg_s.get('seq_len', 128)
        MAX_SEQ_LEN = max(s_long, s_short)
        signals = []
        
        if len(feat_df) >= MAX_SEQ_LEN:
            feat_unix = (feat_df.index.astype(np.int64) // 10**9).tolist()
            features_np = feat_df[feature_cols].values.astype(np.float32)
            
            batch_size = 32
            all_bull = []
            all_bear = []
            for start in range(0, len(features_np) - MAX_SEQ_LEN + 1, batch_size):
                end = min(start + batch_size, len(features_np) - MAX_SEQ_LEN + 1)
                batch_long = [features_np[i + MAX_SEQ_LEN - s_long : i + MAX_SEQ_LEN] for i in range(start, end)]
                batch_short = [features_np[i + MAX_SEQ_LEN - s_short : i + MAX_SEQ_LEN] for i in range(start, end)]
                batch_tensor_l = torch.tensor(np.array(batch_long)).to(_bot_device)
                batch_tensor_s = torch.tensor(np.array(batch_short)).to(_bot_device)
                
                with torch.no_grad():
                    logits_long = _bot_model_long(batch_tensor_l)
                    bull_probs = torch.softmax(logits_long, dim=1)[:, 1].cpu().numpy().tolist()
                    
                    logits_short = _bot_model_short(batch_tensor_s)
                    bear_probs = torch.softmax(logits_short, dim=1)[:, 1].cpu().numpy().tolist()
                    
                    all_bull.extend(bull_probs)
                    all_bear.extend(bear_probs)
            
            # --- Simulate trade lifecycle ---
            position = 'flat'  # flat, long, short
            entry_price = 0
            bars_held = 0
            
            LONG_TP = cfg_l.get('take_profit', 0.0125)
            LONG_SL = cfg_l.get('stop_loss', 0.025)
            LONG_MAX_BARS = cfg_l.get('max_hold_bars', 12)
            SHORT_TP = cfg_s.get('take_profit', 0.015)
            SHORT_SL = cfg_s.get('stop_loss', 0.008)
            SHORT_MAX_BARS = cfg_s.get('max_hold_bars', 8)
            
            for idx in range(len(all_bull)):
                prob_bull = all_bull[idx]
                prob_bear = all_bear[idx]
                bar_idx = MAX_SEQ_LEN - 1 + idx
                
                if bar_idx < len(feat_unix):
                    time_val = feat_unix[bar_idx]
                    close_price = feat_df.iloc[bar_idx]['close'] if 'close' in feat_df.columns else 0
                    high_price = feat_df.iloc[bar_idx]['high'] if 'high' in feat_df.columns else 0
                    low_price = feat_df.iloc[bar_idx]['low'] if 'low' in feat_df.columns else 0
                    
                    import datetime
                    utc_hour = datetime.datetime.utcfromtimestamp(time_val).hour
                    can_trade = True # Trading 24/7 Enabled

                    if position == 'flat':
                        if prob_bull >= 0.60 and can_trade:
                            signals.append({'time': time_val, 'prob': round(prob_bull*100, 6), 'signal': 'BUY', 'price': round(close_price, 2)})
                            position = 'long'
                            entry_price = close_price
                            bars_held = 0
                        
                        # Note: The chart visualizer assumes a single position state for drawing lines,
                        # but we still append the short signal if both fire, or just prioritize.
                        # For chart drawing simplicity, we'll let it switch to short if both fire,
                        # or we can just do independent flat checks. (Since this is just for the chart overlay)
                        if prob_bear >= 0.50 and can_trade and position == 'flat':
                            signals.append({'time': time_val, 'prob': round(prob_bear*100, 6), 'signal': 'SELL', 'price': round(close_price, 2)})
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
                            
        return jsonify({"signals": signals})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "signals": []})

# ---- Background Bot Thread (runs inside same process to save RAM) ----
_bot_thread_started = False

def _run_bot_in_background():
    """Run the ALT-only trading bot in a background thread."""
    import threading
    global _bot_thread_started
    if _bot_thread_started:
        return
    _bot_thread_started = True

    def bot_worker():
        try:
            print("[BOT THREAD] Importing AltOnlyTrader...")
            from trade_live_alt_only import AltOnlyTrader
            print("[BOT THREAD] Creating trader instance...")
            trader = AltOnlyTrader()
            print("[BOT THREAD] Running initial step...")
            trader.step()
            print("[BOT THREAD] Entering main loop...")
            trader.run_forever()
        except Exception as e:
            import traceback
            crash_msg = f"[BOT THREAD] FATAL: {e}\n{traceback.format_exc()}"
            print(crash_msg)
            try:
                with open('alt_bot_crash.log', 'w') as f:
                    f.write(crash_msg)
            except Exception:
                pass

    t = threading.Thread(target=bot_worker, daemon=True)
    t.start()
    print("[BOT THREAD] Background trading bot thread started.")

# Auto-start bot when running under gunicorn (cloud) 
# Check if we're NOT in local dev mode
if os.environ.get("RENDER") or os.environ.get("GUNICORN_CMD_ARGS"):
    _run_bot_in_background()

if __name__ == '__main__':
    # When running locally, also start the bot thread
    _run_bot_in_background()
    print("Dashboard Running! Go to: http://127.0.0.1:5001")
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False, threaded=True)
