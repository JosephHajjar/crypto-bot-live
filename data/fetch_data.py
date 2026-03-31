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

    # Calculate timestamps
    now = datetime.utcnow()
    since_dt = now - timedelta(days=days_back)
    since_ts = int(since_dt.timestamp() * 1000)

    all_ohlcv = []
    
    # Simple progress bar based on expected number of candles approx
    if 'h' in timeframe:
        mins = int(timeframe.replace('h', '')) * 60
    else:
        mins = int(timeframe.replace('m', ''))
    expected_candles = days_back * 24 * 60 // mins
    pbar = tqdm(total=expected_candles)

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
        print("No data fetched.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Remove duplicates if any
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save to CSV
    filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
    filepath = os.path.join(save_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"Successfully saved {len(df)} candles to {filepath}")
    return filepath

if __name__ == '__main__':
    # Default execution: 90 days of 5m BTC/USDT data
    fetch_klines(symbol='BTC/USDT', timeframe='5m', days_back=90)
