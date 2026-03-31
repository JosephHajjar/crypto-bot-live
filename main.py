import os
from data.fetch_data import fetch_klines
from data.feature_engineer import engineer_features
from ml.train import train_model
from engine.backtest import backtest_model

def main():
    print("=== AI Trading Bot Pipeline ===")
    
    # Configuration
    symbol = 'BTC/USDT'
    timeframe = '5m'
    days_back = 30 # Fetch 30 days for this demo, meaning millions of candles might be too much for first time, 30 days is standard 
    seq_length = 60 # Lookback for LSTM (1 hour if 5m candles)
    
    # 1. Fetch
    print("\n[Phase 1] Data Ingestion")
    csv_path = fetch_klines(symbol, timeframe, days_back, save_dir='data_storage')
    
    if not csv_path:
        print("Failed to fetch data.")
        return
        
    # 2. Engineer
    print("\n[Phase 2] Feature Engineering")
    processed_path, _ = engineer_features(csv_path)
    
    # 3. Train
    print("\n[Phase 3] Model Training (GPU)")
    # Normally epochs should be higher, keeping to 10 for quick testing
    train_model(processed_path, epochs=10, seq_length=seq_length)
    
    # 4. Backtest
    print("\n[Phase 4] Backtesting (Out Of Sample)")
    model_path = 'models/best_model.pth'
    if os.path.exists(model_path):
        backtest_model(processed_path, model_path=model_path, seq_length=seq_length)
    else:
        print("Error: No trained model found to backtest.")
        
if __name__ == "__main__":
    main()
