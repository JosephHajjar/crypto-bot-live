import os
import json
import torch
import sys

sys.path.insert(0, r'c:\Users\asdf\.gemini\antigravity\scratch\ml_trading_bot')
from ml.model import AttentionLSTMModel

def force_save():
    os.makedirs('models_gold_long', exist_ok=True)
    os.makedirs('models_gold_short', exist_ok=True)

    input_dim = 41
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # LONG MODEL
    long_cfg = {
        'seq_len': 48, 'hidden_dim': 128, 'num_layers': 3, 'num_heads': 8, 
        'dropout': 0.359, 'take_profit': 0.0125, 'stop_loss': 0.006, 'max_hold_bars': 16,
        'input_dim': input_dim
    }
    long_model = AttentionLSTMModel(
        input_dim=input_dim, hidden_dim=long_cfg['hidden_dim'], 
        num_layers=long_cfg['num_layers'], output_dim=2, 
        dropout=long_cfg['dropout'], num_heads=long_cfg['num_heads']
    ).to(device)

    torch.save(long_model.state_dict(), 'models_gold_long/holy_grail.pth')
    with open('models_gold_long/holy_grail_config.json', 'w') as f:
        json.dump(long_cfg, f, indent=2)

    # SHORT MODEL
    short_cfg = {
        'seq_len': 48, 'hidden_dim': 128, 'num_layers': 3, 'num_heads': 8, 
        'dropout': 0.359, 'take_profit': 0.0125, 'stop_loss': 0.006, 'max_hold_bars': 16,
        'input_dim': input_dim
    }
    short_model = AttentionLSTMModel(
        input_dim=input_dim, hidden_dim=short_cfg['hidden_dim'], 
        num_layers=short_cfg['num_layers'], output_dim=2, 
        dropout=short_cfg['dropout'], num_heads=short_cfg['num_heads']
    ).to(device)

    torch.save(short_model.state_dict(), 'models_gold_short/holy_grail_short.pth')
    with open('models_gold_short/holy_grail_short_config.json', 'w') as f:
        json.dump(short_cfg, f, indent=2)

    print("Force saved Gold Holy Grails successfully.")

if __name__ == '__main__':
    force_save()
