import json
import torch
import pandas as pd
import sys
import os
import glob
sys.path.insert(0, '.')

from optuna_search import backtest_triple_barrier
from ml.model import AttentionLSTMModel

def evaluate_all_models():
    configs = glob.glob("models/*_config.json")
    if not configs:
        print("No models found in models/ directory.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {len(configs)} AI Architectures onto {device}...\n")
    
    df = pd.read_csv("data_storage/BTC_USDT_15m_processed.csv", index_col=0, parse_dates=True)
    
    # We will test the last 30 days for fairness across all of them
    days = 30
    target_bars = days * 24 * 4
    
    results = []
    
    for config_path in configs:
        base_name = os.path.basename(config_path).replace("_config.json", "")
        weight_path = f"models/{base_name}.pth"
        
        if not os.path.exists(weight_path):
            continue
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        model = AttentionLSTMModel(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=2,
            dropout=config['dropout'],
            num_heads=config['num_heads']
        ).to(device)
        
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        except Exception:
            continue
            
        model.eval()
        
        # Calculate start index
        slice_start = len(df) - target_bars - config['seq_len']
        if slice_start < 0:
            slice_start = 0
            
        df_slice = df.iloc[slice_start:]
        
        # Because we only process models that crossed 12.3% validation, 
        # this is exactly what the user ordered
        res = backtest_triple_barrier(df_slice, model, config['seq_len'], device)
        
        # Ignore models that took almost no trades in 30 days
        if res['num_trades'] > 5:
            results.append({
                'name': base_name,
                'roi': res['roi'],
                'sharpe': res['sharpe'],
                'trades': res['num_trades'],
                'wins': res['wins'],
                'losses': res['losses']
            })
            
    # Sort by ROI gracefully
    results.sort(key=lambda x: x['roi'], reverse=True)
    
    print("=" * 70)
    print(f"{'MODEL NAME':<15} | {'30-DAY ROI':<10} | {'SHARPE':<8} | {'TRADES':<8} | {'WIN/LOSS':<10}")
    print("=" * 70)
    
    for r in results:
        roi_str = f"{r['roi']:.2f}%"
        sharpe_str = f"{r['sharpe']:.2f}"
        trades_str = str(r['trades'])
        wl_str = f"{r['wins']}/{r['losses']}"
        print(f"{r['name']:<15} | {roi_str:<10} | {sharpe_str:<8} | {trades_str:<8} | {wl_str:<10}")
        
    print("=" * 70)

if __name__ == "__main__":
    evaluate_all_models()
