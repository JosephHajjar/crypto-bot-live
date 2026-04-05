import ccxt
import pandas as pd
import numpy as np
import torch
import json
import ta
import warnings; warnings.filterwarnings('ignore')
import pytz
from ml.model import AttentionLSTMModel
from data.feature_engineer_btc import get_feature_cols

def get_exact_probs():
    print("Fetching raw exchange tick data for the last few hours...")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '15m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    print("Computing metrics manually (bypassing backward truncation)...")
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    df['EMA_10'] = ta.trend.ema_indicator(close, window=10)
    df['EMA_50'] = ta.trend.ema_indicator(close, window=50)
    df['EMA_200'] = ta.trend.ema_indicator(close, window=200)
    df['Dist_EMA10'] = (close - df['EMA_10']) / df['EMA_10']
    df['Dist_EMA50'] = (close - df['EMA_50']) / df['EMA_50']
    df['Dist_EMA200'] = (close - df['EMA_200']) / df['EMA_200']
    df['RSI_14'] = ta.momentum.rsi(close, window=14) / 100.0
    
    # ATR logic natively used in the model
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean() / close
    
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
    
    macd = ta.trend.MACD(close=close)
    df['MACD'] = macd.macd() / close
    df['MACD_Signal'] = macd.macd_signal() / close
    df['MACD_Hist'] = macd.macd_diff() / close
    
    df['Volume_Ratio'] = volume / volume.rolling(20).mean()
    
    # Session Features
    dt_series = df['timestamp'].dt.tz_localize('UTC')
    df['Hour_Sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
    df['Day_Sin'] = np.sin(2 * np.pi * dt_series.dt.dayofweek / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * dt_series.dt.dayofweek / 7)
    
    # Dummy Macro data just to fill columns (as we don't fetch SPY for rapid tests)
    for c in ['SPY_Close', 'SPY_Dist_EMA50', 'SPY_Volatility', 'GLD_Close', 
              'GLD_Dist_EMA50', 'GLD_Volatility', 'DXY_Close', 'DXY_Dist_EMA50', 
              'DXY_Volatility', 'VIX_Close']:
        df[c] = 0.0

    # Fill NaNs from rolling windows (but not truncating the END)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    print("Loading AI architecture...")
    with open('models/holy_grail_config.json', 'r') as f: cfg=json.load(f)
    m = AttentionLSTMModel(41, 128, 1, 2, 0.1, 2).to('cpu')
    m.load_state_dict(torch.load('models/holy_grail.pth', map_location='cpu', weights_only=True))
    m.eval()

    with open('models_short/holy_grail_short_config.json', 'r') as f: cfg_s=json.load(f)
    ms = AttentionLSTMModel(41, 128, 1, 2, 0.1, 2).to('cpu')
    ms.load_state_dict(torch.load('models_short/holy_grail_short.pth', map_location='cpu', weights_only=True))
    ms.eval()

    cols = get_feature_cols()
    
    # We are missing return distributions, lag returns, etc!
    # The BTC feature_engineer output needs 41 exact columns!
    # Since I don't have all 41 manually computed, I will use `eng` directly BUT 
    # mock `dropna`!
import sys
if __name__ == '__main__':
    get_exact_probs()
