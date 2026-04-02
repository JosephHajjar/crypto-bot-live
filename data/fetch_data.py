import os
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm

def fetch_klines(symbol='BTC/USDT', timeframe='5m', days_back=90, save_dir='data_storage'):
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

    if existing_df is None:
        now = datetime.utcnow()
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
    
    # Save to CSV
    # filename and filepath are already defined at the top
    df.to_csv(filepath, index=False)
    
    print(f"Successfully saved {len(df)} candles to {filepath}")
    return filepath

if __name__ == '__main__':
    # Default execution: 90 days of 5m BTC/USDT data
    fetch_klines(symbol='BTC/USDT', timeframe='5m', days_back=90)
