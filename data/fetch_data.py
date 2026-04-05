import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import yfinance as yf

def fetch_klines(symbol='BTC/USDT', timeframe='5m', days_back=90, save_dir='data_storage', fetch_macro=False):
    """
    Fetches historical OHLCV data from Binance and saves it to a CSV.
    Uses pagination since Binance limits to 1000 candles per request.
    """
    print(f"Fetching {timeframe} data for {symbol} for the last {days_back} days...")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })

    filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
    filepath = os.path.join(save_dir, filename)
    
    existing_df = None
    if os.path.exists(filepath):
        print(f"Loading existing data from {filepath} to resume fetching...")
        try:
            existing_df = pd.read_csv(filepath)
            if not existing_df.empty and 'timestamp' in existing_df.columns:
                # Convert timestamps safely
                if existing_df['timestamp'].dtype == 'O':
                    existing_ts = pd.to_datetime(existing_df['timestamp']).astype(np.int64) // 10**6
                else:
                    existing_ts = pd.to_datetime(existing_df['timestamp'], unit='s' if existing_df['timestamp'].max() < 1e11 else 'ms').astype(np.int64) // 10**6
                
                last_ts = int(existing_ts.max())
                since_ts = last_ts + 1  # start from the very next ms
                print(f"Resuming fetch from {pd.to_datetime(last_ts, unit='ms')}")
            else:
                existing_df = None
        except Exception as e:
            print(f"Could not read existing file: {e}. Fetching {days_back} days.")
            existing_df = None

    now = datetime.utcnow()
    if existing_df is None:
        since_dt = now - timedelta(days=days_back)
        since_ts = int(since_dt.timestamp() * 1000)

    # Simple progress bar based on approx expected number of new candles
    now_ts = int(datetime.utcnow().timestamp() * 1000)
    if 'h' in timeframe:
        mins = int(timeframe.replace('h', '')) * 60
    else:
        mins = int(timeframe.replace('m', ''))
        
    expected_candles = max(1, (now_ts - since_ts) // (mins * 60 * 1000))
    pbar = tqdm(total=expected_candles)
    all_ohlcv = []

    while since_ts < int(now.timestamp() * 1000):
        try:
            # fetch ohlcv
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=1000)
            
            if len(ohlcv) == 0:
                break
                
            all_ohlcv.extend(ohlcv)
            
            # Update 'since' to the last timestamp + 1 ms to avoid duplication
            since_ts = ohlcv[-1][0] + 1
            
            pbar.update(len(ohlcv))
            # Sleep slightly to respect rate limits
            time.sleep(exchange.rateLimit / 1000)
            
        except ccxt.NetworkError as e:
            print(f"Network error: {e}. Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    pbar.close()

    if not all_ohlcv:
        print("No new data fetched.")
        return filepath

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    if existing_df is not None:
        if existing_df['timestamp'].dtype == 'O':
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
        df = pd.concat([existing_df, df], ignore_index=True)
    
    # Remove duplicates if any and sort exactly
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    df.sort_values('timestamp', inplace=True)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    if fetch_macro:
        print("Fetching macro context (DXY, VIX) via yfinance...")
        try:
            start_dt = df['timestamp'].min() - pd.Timedelta(days=5) # Buffer for ffill
            end_dt = df['timestamp'].max() + pd.Timedelta(days=1)
            
            # yfinance limits 1h data to max 730 days ago
            min_allowed_start = pd.Timestamp(datetime.utcnow() - timedelta(days=729))
            # Start dt must not be older than min_allowed_start
            yf_start = max(start_dt, min_allowed_start)
            
            dxy_data = yf.download('DX-Y.NYB', start=yf_start, end=end_dt, interval='1h', progress=False)
            vix_data = yf.download('^VIX', start=yf_start, end=end_dt, interval='1h', progress=False)
            
            # Extract close prices safely considering recent yfinance MultiIndex changes
            if isinstance(dxy_data.columns, pd.MultiIndex):
                dxy_close = dxy_data['Close']['DX-Y.NYB']
            else:
                dxy_close = dxy_data['Close']
                
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_close = vix_data['Close']['^VIX']
            else:
                vix_close = vix_data['Close']
                
            dxy_df = dxy_close.to_frame(name='DXY').dropna()
            vix_df = vix_close.to_frame(name='VIX').dropna()
            
            if dxy_df.empty:
                df['DXY'] = 100.0
            else:
                dxy_df.index = dxy_df.index.tz_localize(None)
                df = pd.merge_asof(df, dxy_df, left_on='timestamp', right_index=True, direction='backward')
                df['DXY'] = df['DXY'].ffill().bfill()
                
            if vix_df.empty:
                df['VIX'] = 20.0
            else:
                vix_df.index = vix_df.index.tz_localize(None)
                df = pd.merge_asof(df, vix_df, left_on='timestamp', right_index=True, direction='backward')
                df['VIX'] = df['VIX'].ffill().bfill()
                
        except Exception as e:
            print(f"Warning: Failed to fetch macro data: {e}. Filling with defaults.")
            df['DXY'] = 100.0
            df['VIX'] = 20.0
    
    # Save to CSV
    # filename and filepath are already defined at the top
    df.to_csv(filepath, index=False)
    
    print(f"Successfully saved {len(df)} candles to {filepath}")
    return filepath

if __name__ == '__main__':
    # Default execution: 90 days of 5m BTC/USDT data
    fetch_klines(symbol='BTC/USDT', timeframe='5m', days_back=90)
