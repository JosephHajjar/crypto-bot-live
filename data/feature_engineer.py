import pandas as pd
import ta
import numpy as np
import os

def _resample_ohlcv(df, rule):
    """Resample 15m OHLCV data to a higher timeframe (e.g., '1h', '4h')."""
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return resampled

def _compute_indicators(df, prefix=''):
    """Compute technical indicators on a dataframe. Returns new columns as a dict."""
    cols = {}
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Trend
    cols[f'{prefix}SMA_10'] = ta.trend.sma_indicator(close, window=10)
    cols[f'{prefix}SMA_50'] = ta.trend.sma_indicator(close, window=50)
    cols[f'{prefix}EMA_12'] = ta.trend.ema_indicator(close, window=12)
    cols[f'{prefix}EMA_26'] = ta.trend.ema_indicator(close, window=26)
    
    # RSI
    cols[f'{prefix}RSI_14'] = ta.momentum.rsi(close, window=14)
    
    # MACD
    cols[f'{prefix}MACD'] = ta.trend.macd(close, window_slow=26, window_fast=12)
    cols[f'{prefix}MACD_hist'] = ta.trend.macd_diff(close, window_slow=26, window_fast=12, window_sign=9)
    cols[f'{prefix}MACD_signal'] = ta.trend.macd_signal(close, window_slow=26, window_fast=12, window_sign=9)
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    cols[f'{prefix}BB_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg() * 100
    cols[f'{prefix}BB_pct'] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
    
    # ATR
    cols[f'{prefix}ATR_14'] = ta.volatility.average_true_range(high=high, low=low, close=close, window=14)
    
    # Volume ratio  
    vol_sma = ta.trend.sma_indicator(volume, window=20)
    cols[f'{prefix}Vol_Ratio'] = volume / (vol_sma + 1e-8)
    
    return cols

def _triple_barrier_label(df, take_profit=0.015, stop_loss=0.0075, max_hold_bars=16):
    """
    Triple Barrier Method labeling.
    For each bar, look forward up to max_hold_bars:
      - If price hits take_profit first -> label 1 (profitable trade)
      - If price hits stop_loss first -> label 0 (losing trade)
      - If neither hit within max_hold_bars -> label 0 (no clear signal)
    """
    close = df['close'].values
    labels = np.zeros(len(close), dtype=int)
    
    for i in range(len(close) - max_hold_bars):
        entry = close[i]
        tp_price = entry * (1 + take_profit)
        sl_price = entry * (1 - stop_loss)
        
        for j in range(1, max_hold_bars + 1):
            future_high = df['high'].values[i + j]
            future_low = df['low'].values[i + j]
            
            # Check stop loss first (conservative — assumes worst case within candle)
            if future_low <= sl_price:
                labels[i] = 0
                break
            # Check take profit
            if future_high >= tp_price:
                labels[i] = 1
                break
        # If loop completes without break, label stays 0
    
    return labels

def _triple_barrier_label_short(df, take_profit=0.015, stop_loss=0.0075, max_hold_bars=16):
    """
    Triple Barrier Method labeling for SCHORTS.
    For each bar, look forward up to max_hold_bars:
      - If price hits take_profit first (down) -> label 1 (profitable short)
      - If price hits stop_loss first (up) -> label 0 (losing short)
      - If neither hit within max_hold_bars -> label 0
    """
    close = df['close'].values
    labels = np.zeros(len(close), dtype=int)
    
    for i in range(len(close) - max_hold_bars):
        entry = close[i]
        tp_price = entry * (1 - take_profit)
        sl_price = entry * (1 + stop_loss)
        
        for j in range(1, max_hold_bars + 1):
            future_high = df['high'].values[i + j]
            future_low = df['low'].values[i + j]
            
            # Check stop loss first
            if future_high >= sl_price:
                labels[i] = 0
                break
            # Check take profit
            if future_low <= tp_price:
                labels[i] = 1
                break
    
    return labels

def engineer_features(csv_path, take_profit=0.015, stop_loss=0.0075, max_hold_bars=16, mode='long'):
    """
    Multi-timeframe feature engineering with Triple Barrier labeling.
    
    Reads 15m OHLCV data, computes indicators at 15m/1h/4h timeframes,
    adds microstructure features, and labels using Triple Barrier method.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # ================================================================
    # 1. BASE 15-MINUTE INDICATORS
    # ================================================================
    print("Computing 15m indicators...")
    base_indicators = _compute_indicators(df, prefix='')
    for col_name, col_data in base_indicators.items():
        df[col_name] = col_data
    
    # ================================================================
    # 2. MULTI-TIMEFRAME INDICATORS (1h, 4h)
    # ================================================================
    print("Computing 1h indicators...")
    df_1h = _resample_ohlcv(df[['open', 'high', 'low', 'close', 'volume']], '1h')
    indicators_1h = _compute_indicators(df_1h, prefix='1h_')
    df_1h_features = pd.DataFrame(indicators_1h, index=df_1h.index)
    # Forward-fill 1h features onto 15m index (no look-ahead: each 15m bar gets the LAST completed 1h value)
    df = df.join(df_1h_features, how='left')
    df[list(indicators_1h.keys())] = df[list(indicators_1h.keys())].ffill()
    
    print("Computing 4h indicators...")
    df_4h = _resample_ohlcv(df[['open', 'high', 'low', 'close', 'volume']], '4h')
    indicators_4h = _compute_indicators(df_4h, prefix='4h_')
    df_4h_features = pd.DataFrame(indicators_4h, index=df_4h.index)
    df = df.join(df_4h_features, how='left')
    df[list(indicators_4h.keys())] = df[list(indicators_4h.keys())].ffill()
    
    # ================================================================
    # 3. MICROSTRUCTURE FEATURES
    # ================================================================
    print("Computing microstructure features...")
    
    # Percentage Returns
    df['Returns'] = df['close'].pct_change()
    
    # Realized Volatility (rolling std of returns)
    df['RealVol_12'] = df['Returns'].rolling(12).std()  # 3-hour realized vol
    df['RealVol_48'] = df['Returns'].rolling(48).std()  # 12-hour realized vol
    df['Vol_Regime'] = df['RealVol_12'] / (df['RealVol_48'] + 1e-10)  # vol expansion/contraction
    
    # Multi-horizon momentum
    df['Mom_4'] = df['close'].pct_change(4)    # 1-hour momentum
    df['Mom_16'] = df['close'].pct_change(16)  # 4-hour momentum
    df['Mom_48'] = df['close'].pct_change(48)  # 12-hour momentum
    df['Mom_96'] = df['close'].pct_change(96)  # 24-hour momentum
    
    # Volume Imbalance (approximation using candle body position within range)
    body = df['close'] - df['open']
    wick_range = df['high'] - df['low']
    df['Vol_Imbalance'] = body / (wick_range + 1e-10)  # +1 = full bullish, -1 = full bearish
    
    # Candle body ratio (how much of the candle is body vs wick)
    df['Body_Ratio'] = abs(body) / (wick_range + 1e-10)
    
    # Time-of-day encoding (cyclical — crypto has daily session patterns)
    hour = df.index.hour + df.index.minute / 60.0
    df['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Day-of-week encoding (cyclical)
    dow = df.index.dayofweek
    df['DOW_sin'] = np.sin(2 * np.pi * dow / 7)
    df['DOW_cos'] = np.cos(2 * np.pi * dow / 7)
    
    # Price distance from key levels
    df['Dist_SMA50'] = (df['close'] - df['SMA_50']) / (df['SMA_50'] + 1e-10)
    
    # ================================================================
    # 4. TRIPLE BARRIER LABELING
    # ================================================================
    print(f"Labeling with Triple Barrier (TP={take_profit*100:.1f}%, SL={stop_loss*100:.2f}%, MaxBars={max_hold_bars}, mode={mode})...")
    if mode == 'short':
        df['Target'] = _triple_barrier_label_short(df, take_profit, stop_loss, max_hold_bars)
    else:
        df['Target'] = _triple_barrier_label(df, take_profit, stop_loss, max_hold_bars)
    
    # ================================================================
    # 5. CLEANUP AND NORMALIZE
    # ================================================================
    df = df.dropna()
    
    # Remove the last max_hold_bars rows (they can't have proper labels)
    df = df.iloc[:-max_hold_bars]
    
    # Define all feature columns
    feature_cols = get_feature_cols()
    
    # Verify all features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"WARNING: Missing features: {missing}")
        feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Z-score normalize using ONLY training data (first 70%)
    print("Normalizing features (train-only statistics)...")
    train_end_idx = int(len(df) * 0.7)
    train_slice = df[feature_cols].iloc[:train_end_idx]
    
    feature_means = train_slice.mean()
    feature_stds = train_slice.std().replace(0, 1)
    
    df[feature_cols] = (df[feature_cols] - feature_means) / feature_stds
    
    # Save
    if mode == 'short':
        save_path = csv_path.replace('.csv', '_short_processed.csv')
        scaler_save = csv_path.replace('.csv', '_short_scaler.json')
    else:
        save_path = csv_path.replace('.csv', '_processed.csv')
        scaler_save = csv_path.replace('.csv', '_scaler.json')
        
    df.to_csv(save_path)
    
    # Save scalars to disk
    import json
    stats = {
        'mean': feature_means.to_dict(),
        'std': feature_stds.to_dict()
    }
    with open(scaler_save, 'w') as f:
        json.dump(stats, f)
        
    # Print class distribution
    target_dist = df['Target'].value_counts()
    pct_positive = target_dist.get(1, 0) / len(df) * 100
    print(f"Saved {len(df)} samples to {save_path}")
    print(f"Target distribution: {target_dist.to_dict()} ({pct_positive:.1f}% positive)")
    
    return save_path, stats

def compute_live_features(df, scaler_path="data_storage/BTC_USDT_15m_scaler.json"):
    """
    Computes indicators on live streaming DataFrame.
    Crucially drops future-leakage (No Triple-Barrier labeling) and retains the most recent candles for actual inference.
    """
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
    # 1. BASE 15-MINUTE INDICATORS
    base_indicators = _compute_indicators(df, prefix='')
    for col_name, col_data in base_indicators.items():
        df[col_name] = col_data
        
    # 2. MULTI-TIMEFRAME INDICATORS (1h, 4h)
    df_1h = _resample_ohlcv(df[['open', 'high', 'low', 'close', 'volume']], '1h')
    indicators_1h = _compute_indicators(df_1h, prefix='1h_')
    df_1h_features = pd.DataFrame(indicators_1h, index=df_1h.index)
    df = df.join(df_1h_features, how='left')
    df[list(indicators_1h.keys())] = df[list(indicators_1h.keys())].ffill()
    
    df_4h = _resample_ohlcv(df[['open', 'high', 'low', 'close', 'volume']], '4h')
    indicators_4h = _compute_indicators(df_4h, prefix='4h_')
    df_4h_features = pd.DataFrame(indicators_4h, index=df_4h.index)
    df = df.join(df_4h_features, how='left')
    df[list(indicators_4h.keys())] = df[list(indicators_4h.keys())].ffill()
    
    # 3. MICROSTRUCTURE FEATURES
    df['Returns'] = df['close'].pct_change()
    df['RealVol_12'] = df['Returns'].rolling(12).std()
    df['RealVol_48'] = df['Returns'].rolling(48).std()
    df['Vol_Regime'] = df['RealVol_12'] / (df['RealVol_48'] + 1e-10)
    
    df['Mom_4'] = df['close'].pct_change(4)
    df['Mom_16'] = df['close'].pct_change(16)
    df['Mom_48'] = df['close'].pct_change(48)
    df['Mom_96'] = df['close'].pct_change(96)
    
    body = df['close'] - df['open']
    wick_range = df['high'] - df['low']
    df['Vol_Imbalance'] = body / (wick_range + 1e-10)
    df['Body_Ratio'] = abs(body) / (wick_range + 1e-10)
    
    hour = df.index.hour + df.index.minute / 60.0
    df['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    dow = df.index.dayofweek
    df['DOW_sin'] = np.sin(2 * np.pi * dow / 7)
    df['DOW_cos'] = np.cos(2 * np.pi * dow / 7)
    
    df['Dist_SMA50'] = (df['close'] - df['SMA_50']) / (df['SMA_50'] + 1e-10)
    
    df = df.dropna()
    
    # NORMALIZATION
    feature_cols = get_feature_cols()
    
    import json
    with open(scaler_path, 'r') as f:
        scaler = json.load(f)
        
    for col in feature_cols:
        if col in df.columns and col in scaler['mean']:
            df[col] = (df[col] - scaler['mean'][col]) / scaler['std'][col]
            
    return df

def get_feature_cols():
    """Returns the canonical list of feature columns used everywhere."""
    base = ['Returns', 'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI_14', 'MACD', 'MACD_hist', 'MACD_signal',
            'BB_width', 'BB_pct', 'ATR_14', 'Vol_Ratio']
    
    htf_1h = [f'1h_{x}' for x in ['RSI_14', 'MACD', 'MACD_hist', 'BB_width', 'BB_pct', 'ATR_14', 'Vol_Ratio']]
    htf_4h = [f'4h_{x}' for x in ['RSI_14', 'MACD', 'MACD_hist', 'BB_width', 'BB_pct', 'ATR_14', 'Vol_Ratio']]
    
    micro = ['RealVol_12', 'RealVol_48', 'Vol_Regime',
             'Mom_4', 'Mom_16', 'Mom_48', 'Mom_96',
             'Vol_Imbalance', 'Body_Ratio',
             'Hour_sin', 'Hour_cos', 'DOW_sin', 'DOW_cos',
             'Dist_SMA50']
    
    return base + htf_1h + htf_4h + micro

if __name__ == "__main__":
    dummy_path = "data_storage/BTC_USDT_15m.csv"
    if os.path.exists(dummy_path):
        engineer_features(dummy_path)
    else:
        print(f"Run fetch_data.py first. Could not find {dummy_path}")
