import os
import torch
import json
import pandas as pd
import numpy as np

from ml.model import LSTMTradingModel

def analyze_model(model_name="best_so_far"):
    model_path = f"models/{model_name}.pth"
    config_path = f"models/{model_name}_config.json"
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"Cannot find both {model_path} and {config_path}. Waiting for a high score model constraint to be met first.")
        return
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    seq_len = config['seq_len']
    hidden_dim = config['hidden_dim']
    dropout = config['dropout']
    
    print(f"=== Deep Analyis for {model_name} ===")
    print(f"Config: Sequence Length: {seq_len}, Hidden Dim: {hidden_dim}, Dropout: {dropout}")
    
    processed_path = "data_storage/BTC_USDT_5m_processed.csv"
    if not os.path.exists(processed_path):
        print("Dataset not found. Please ensure the dataset exists.")
        return
        
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    df_test = df.iloc[val_end:].copy()
    
    feature_cols = [
        'Returns', 'SMA_10', 'SMA_50', 'RSI_14', 
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
        'ATR_14', 'Vol_Ratio'
    ]
    
    input_dim = len(feature_cols)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMTradingModel(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        num_layers=2, 
        output_dim=2, 
        dropout=dropout
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Evaluating {model_name} on {len(df_test)} Unseen Test Data rows...")
    
    X_val = df_test[feature_cols].values
    close_prices = df_test['close'].values
    timestamps = df_test.index
    
    all_seqs = [X_val[i-seq_len : i] for i in range(seq_len, len(X_val) - 1)]
    if len(all_seqs) == 0:
        print("Not enough test data.")
        return
        
    seq_tensor = torch.tensor(np.array(all_seqs), dtype=torch.float32).to(device)
    
    signals = []
    with torch.no_grad():
        batch_size = 2048
        for i in range(0, len(seq_tensor), batch_size):
            batch = seq_tensor[i : i+batch_size]
            out = model(batch)
            _, pred = torch.max(out, 1)
            signals.extend(pred.cpu().tolist())
            
    fee_pct = 0.001
    
    # Store decisions
    trade_log = []
    
    for idx, i in enumerate(range(seq_len, len(X_val) - 1)):
        signal = signals[idx]
        if signal == 1:
            # Bot took a trade exactly at this candle close!
            current_close = close_prices[i]
            # Next Candle Return
            next_return = (close_prices[i+1] - close_prices[i]) / close_prices[i]
            
            # Record what the indicators were exactly at the time of the trade
            row_data = df_test.iloc[i].to_dict()
            row_data['Timestamp'] = timestamps[i]
            row_data['Signal'] = "BUY_AND_HOLD_1_PERIOD"
            row_data['Candle_Close'] = current_close
            row_data['Next_Return_Pct'] = next_return * 100
            row_data['Fee_Impact_Pct'] = fee_pct * 100 * 2 # In and Out
            
            net_profit_pct = (next_return - (fee_pct * 2)) * 100
            row_data['Net_Profit_Pct'] = net_profit_pct
            row_data['Win'] = 1 if net_profit_pct > 0 else 0
            
            trade_log.append(row_data)
            
    if len(trade_log) == 0:
        print("Model took precisely 0 trades on the Unseen Test Data. It was likely biased towards Class 0.")
        return
        
    results_df = pd.DataFrame(trade_log)
    output_csv = f'models/{model_name}_insights.csv'
    results_df.to_csv(output_csv, index=False)
    
    win_rate = results_df['Win'].mean() * 100
    avg_gain = results_df.loc[results_df['Win'] == 1, 'Net_Profit_Pct'].mean()
    avg_loss = results_df.loc[results_df['Win'] == 0, 'Net_Profit_Pct'].mean()
    total_net = results_df['Net_Profit_Pct'].sum()
    
    print("\n" + "="*50)
    print("=== MODEL DEEP INSIGHT REPORT ===")
    print("="*50)
    print(f"Total Trades Taken on Unseen Data: {len(results_df)}")
    print(f"Win Rate:               {win_rate:.2f}%")
    print(f"Average Winning Trade:  {avg_gain if not pd.isna(avg_gain) else 0:.4f}%")
    print(f"Average Losing Trade:   {avg_loss if not pd.isna(avg_loss) else 0:.4f}%")
    print(f"Cumulative Net Returns: {total_net:.2f}% (Compounding un-applied)")
    print("\n--- Average Indicator Status During Trade Initiation ---")
    print(f"RSI 14 (Z-Scored): {results_df['RSI_14'].mean():.4f}")
    print(f"MACD (Z-Scored):   {results_df['MACD_12_26_9'].mean():.4f}")
    print(f"Volatility Ratio:  {results_df['Vol_Ratio'].mean():.4f}")
    print("======================================================")
    print(f"Extensive CSV trade-by-trade breakdown saved to: {output_csv}")
    print("Open the CSV to see exactly what reality looked like when the AI fired every signal.")
    
if __name__ == "__main__":
    analyze_model("best_so_far")
