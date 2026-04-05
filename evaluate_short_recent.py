import os
import json
import torch
import pandas as pd
import glob
from optuna_search_short import backtest_triple_barrier_short
from ml.model import AttentionLSTMModel
from ml.dataset import TimeSeriesDataset

def main():
    model_dir = "models_short"
    data_file = "data_storage/BTC_USDT_15m_short_processed.csv"
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
        
    print(f"Loading data from {data_file}...")
    full_df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    configs = glob.glob(f"{model_dir}/*_config.json")
    if not configs:
        print(f"No models found in {model_dir}")
        return
        
    print(f"Evaluating {len(configs)} short models over recent timeframes...")
    
    results = []
    
    for config_file in sorted(configs):
        with open(config_file, 'r') as f:
            cfg = json.load(f)
            
        model_name = os.path.basename(config_file).replace('_config.json', '')
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        
        if not os.path.exists(model_path):
            continue
            
        # Build model from config
        model = AttentionLSTMModel(
            input_dim=cfg['input_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            output_dim=2,
            dropout=cfg['dropout'],
            num_heads=cfg['num_heads']
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Test windows (in 15m candles: 1d=96, 3d=288, 7d=672, 14d=1344)
        windows = {'1d': 96, '3d': 288, '7d': 672, '14d': 1344}
        
        model_res = {'name': model_name, 'tp': cfg.get('take_profit', 0.015), 'sl': cfg.get('stop_loss', 0.0075)}
        
        for win_name, win_bars in windows.items():
            # Include sequence length buffer to ensure we get exactly win_bars of inference
            # To get inference on the last `win_bars` candles, we need `seq_len` previous candles as context
            slice_bars = win_bars + cfg['seq_len'] + cfg.get('max_hold_bars', 16) + 10
            
            if len(full_df) > slice_bars:
                df_slice = full_df.iloc[-slice_bars:]
            else:
                df_slice = full_df
                
            res = backtest_triple_barrier_short(
                df_slice, 
                model, 
                cfg['seq_len'], 
                device, 
                tp=cfg.get('take_profit', 0.015),
                sl=cfg.get('stop_loss', 0.0075),
                max_bars=cfg.get('max_hold_bars', 16)
            )
            
            # Format nicely
            roi = res['roi']
            annual = res['annual_roi']
            sharpe = res['sharpe']
            winrate = (res['wins']/res['num_trades']*100) if res['num_trades']>0 else 0
            model_res[win_name] = f"ROI: {roi:5.2f}% | W: {winrate:4.1f}% | Tr: {res['num_trades']}"
            
        results.append(model_res)
        
    # Print results table
    print("\n" + "="*110)
    print(f"{'MODEL':<20} | {'TP / SL':<15} | {'LAST 1 DAY':<20} | {'LAST 3 DAYS':<20} | {'LAST 7 DAYS':<20}")
    print("="*110)
    for r in results:
        tpsl = f"{r['tp']*100:.1f}/{r['sl']*100:.1f}%"
        print(f"{r['name']:<20} | {tpsl:<15} | {r.get('1d', 'N/A'):<20} | {r.get('3d', 'N/A'):<20} | {r.get('7d', 'N/A'):<20}")
    print("="*110)
    
if __name__ == "__main__":
    main()
