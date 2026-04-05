import pandas as pd
import ta
import numpy as np
import os
import json
from numba import njit
import sys

def precompute_static_features(csv_path):
    """
    Computes all static features (EMAs, session times, higher timeframe lookups, DXY/VIX)
    and saves them to a base processed file to be loaded quickly during Optuna trials.
    """
    print(f"Loading raw data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    volume = df['volume']

    # 1. EMAs and distances
    df['EMA_10'] = ta.trend.ema_indicator(close, window=10)
    df['EMA_50'] = ta.trend.ema_indicator(close, window=50)
    df['EMA_200'] = ta.trend.ema_indicator(close, window=200)

    df['Dist_EMA10'] = (close - df['EMA_10']) / df['EMA_10']
    df['Dist_EMA50'] = (close - df['EMA_50']) / df['EMA_50']
    df['Dist_EMA200'] = (close - df['EMA_200']) / df['EMA_200']
    df['EMA_10_50_Ratio'] = df['EMA_10'] / df['EMA_50']

    # 2. MACD
    df['MACD'] = ta.trend.macd(close)
    df['MACD_signal'] = ta.trend.macd_signal(close)
    df['MACD_hist'] = ta.trend.macd_diff(close)

    # 3. Bollinger Bands
    bb = ta.volatility.BollingerBands(close)
    df['BB_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df['BB_pct'] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)

    # 4. Volume Delta
    df['Volume_Delta'] = volume.diff()

    # 5. Previous Return & Volatility
    df['Return'] = close.pct_change()
    df['Prev_Return'] = df['Return'].shift(1)
    df['Rolling_Vol_20'] = df['Return'].rolling(20).std()

    # 6. Session Data
    # Assuming timezone is UTC
    hour = df.index.hour
    df['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * hour / 24)

    dow = df.index.dayofweek
    df['DOW_sin'] = np.sin(2 * np.pi * dow / 7)
    df['DOW_cos'] = np.cos(2 * np.pi * dow / 7)

    # London: 8-16 UTC, NY: 13-21 UTC
    df['Is_London'] = ((hour >= 8) & (hour < 16)).astype(int)
    df['Is_NY'] = ((hour >= 13) & (hour < 21)).astype(int)

    # 7. Higher Timeframe Resampling (4H, Daily)
    def resample_htf(rule, prefix):
        htf = df[['open', 'high', 'low', 'close', 'volume']].resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        htf_features = pd.DataFrame(index=htf.index)
        # Macro context: distance from 200 EMA
        htf_features[f'{prefix}_EMA_200'] = ta.trend.ema_indicator(htf['close'], window=200)
        htf_features[f'{prefix}_Dist_EMA200'] = (htf['close'] - htf_features[f'{prefix}_EMA_200']) / htf_features[f'{prefix}_EMA_200']
        return htf_features

    df_4h_feat = resample_htf('4h', '4h')
    df_1d_feat = resample_htf('1d', '1d')

    # Join lower frequency onto our 1H dataframe (forward filled to avoid look ahead logic)
    df = df.join(df_4h_feat, how='left').join(df_1d_feat, how='left')
    df['4h_Dist_EMA200'] = df['4h_Dist_EMA200'].ffill()
    df['1d_Dist_EMA200'] = df['1d_Dist_EMA200'].ffill()

    # Drop early rows with NaNs (EMA 200 inherently removes the first 200 intervals)
    df = df.dropna(subset=['EMA_200', '4h_Dist_EMA200', '1d_Dist_EMA200'])

    # Default 'DXY' and 'VIX' gracefully if fetch_data.py didn't inject them
    if 'DXY' not in df.columns:
        df['DXY'] = 100.0
    if 'VIX' not in df.columns:
        df['VIX'] = 20.0

    # Clean extreme inf values occasionally produced by math divisions
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    out_path = csv_path.replace('.csv', '_static.csv')
    df.to_csv(out_path)
    print(f"Saved {len(df)} static processed features to {out_path}")
    return out_path

@njit
def _compute_3class_labels(close, high, low, atr, threshold, horizon):
    """
    Classes:
    0 = Neither / Flat (Skip trade)
    1 = Long (+1: Rises > threshold * ATR)
    2 = Short (-1: Falls > threshold * ATR) 
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int64)
    
    for i in range(n - horizon):
        entry_price = close[i]
        barrier = atr[i] * threshold
        up_barrier = entry_price + barrier
        dn_barrier = entry_price - barrier
        
        for j in range(1, horizon + 1):
            f_high = high[i + j]
            f_low = low[i + j]
            
            # First touch logic
            # If high hits up_barrier AND low hits dn_barrier in the exact same hourly candle
            if f_high >= up_barrier and f_low <= dn_barrier:
                # Ambiguous noise: ignore trade
                labels[i] = 0
                break
                
            elif f_high >= up_barrier:
                labels[i] = 1 # Long
                break
                
            elif f_low <= dn_barrier:
                labels[i] = 2 # Short
                break
                
    return labels

def dynamic_features_and_labels(df, atr_period, rsi_period, label_horizon, label_threshold):
    """
    Computes dynamic features natively inside the Optuna Trial loop extremely fast!
    """
    df = df.copy()
    
    # 1. Dynamic TSI/RSI
    df['RSI'] = ta.momentum.rsi(df['close'], window=rsi_period)
    
    # 2. Dynamic ATR
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=atr_period)
    df['ATR_Norm'] = df['ATR'] / df['close']  # Normalized by price scaling

    # 3. Target Barrier Computation
    df['Target'] = _compute_3class_labels(
        df['close'].values, 
        df['high'].values, 
        df['low'].values, 
        df['ATR'].values,
        label_threshold,
        label_horizon
    )

    df = df.dropna()
    # Remove last incomplete futures
    df = df.iloc[:-label_horizon]
    return df

def get_feature_cols():
    """Canonical feature lookup for the Gold 1H Strategy Model."""
    return [
        'EMA_10', 'EMA_50', 'EMA_200', 'Dist_EMA10', 'Dist_EMA50', 'Dist_EMA200', 'EMA_10_50_Ratio',
        'MACD', 'MACD_signal', 'MACD_hist', 'BB_width', 'BB_pct', 'Volume_Delta',
        'Return', 'Prev_Return', 'Rolling_Vol_20', 
        'Hour_sin', 'Hour_cos', 'DOW_sin', 'DOW_cos', 'Is_London', 'Is_NY',
        '4h_Dist_EMA200', '1d_Dist_EMA200', 'DXY', 'VIX',
        'RSI', 'ATR_Norm'
    ]

if __name__ == "__main__":
    dummy_path = "data_storage/PAXG_USDT_1h.csv"
    if os.path.exists(dummy_path):
        precompute_static_features(dummy_path)
    else:
        print(f"File {dummy_path} missing.")
