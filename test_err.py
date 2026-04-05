import torch
import numpy as np
import json
import sys
sys.path.insert(0, '.')
from ml.model import AttentionLSTMModel
import pandas as pd
from data.feature_engineer_btc import compute_live_features, get_feature_cols
import requests

def test_dashboard_api():
    try:
        res = requests.get("http://127.0.0.1:5001/api/bot_signals?limit=1", timeout=5).json()
        print(res)
    except Exception as e:
        print(e)
        
if __name__ == "__main__":
    test_dashboard_api()
