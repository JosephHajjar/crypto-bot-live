import os, json, time, datetime
import requests
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ml.model import AttentionLSTMModel
from data.feature_engineer_btc import compute_live_features, get_feature_cols

# ------------------- Config -------------------
CONFIG = {
    "seq_len": 128,
    "hidden_dim": 64,
    "num_layers": 1,
    "num_heads": 4,
    "dropout": 0.23,
    "input_dim": 41,
    "take_profit": 0.0125,
    "stop_loss": 0.025,
    "max_hold_bars": 12,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "max_epochs": 30,
    "patience": 5,
    "train_days": 30  # how many days of 5-min data to pull
}

MODEL_DIR = "models_5m"
os.makedirs(MODEL_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(MODEL_DIR, "holy_grail_5m_config.json")
with open(CONFIG_PATH, "w") as f:
    json.dump(CONFIG, f, indent=2)

# ------------------- Data fetch -------------------
def fetch_5m_data(days: int = 30):
    symbol = "BTCUSDT"
    interval = "5m"
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - days * 24 * 60 * 60 * 1000
    url = (
        f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}"
        f"&startTime={start_ts}&endTime={end_ts}&limit=1000"
    )
    all_rows = []
    while True:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        last_ts = data[-1][0]
        if last_ts >= end_ts:
            break
        url = (
            f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}"
            f"&startTime={last_ts+1}&endTime={end_ts}&limit=1000"
        )
    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "ignore1", "ignore2", "ignore3", "ignore4", "ignore5", "ignore6"],
    )
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df

# ------------------- Training utilities -------------------

def prepare_dataset(df: pd.DataFrame, seq_len: int):
    scaler_path = "data_storage/BTC_USDT_15m_scaler.json"
    feat_df = compute_live_features(df, scaler_path)
    cols = get_feature_cols()
    X = feat_df[cols].values.astype(np.float32)
    # Simple proxy labels: price up in next candle -> long (1), else short (0)
    future_close = df["close"].shift(-1).values
    price_diff = future_close - df["close"].values
    y = np.where(price_diff > 0, 1, 0)
    seq_X = []
    seq_y = []
    for i in range(len(X) - seq_len):
        seq_X.append(X[i:i+seq_len])
        seq_y.append(y[i+seq_len])
    seq_X = np.stack(seq_X)
    seq_y = np.array(seq_y)
    return seq_X, seq_y

def train_model(model: nn.Module, train_loader, val_loader, device, cfg):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        print(f"Epoch {epoch:02d} - Train loss: {epoch_loss:.4f} - Val loss: {val_loss:.4f} - Val acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print("Early stopping triggered")
                break
    model.load_state_dict(best_state)
    return model

# ------------------- Main -------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    df = fetch_5m_data(days=CONFIG["train_days"])
    print(f"Fetched {len(df)} rows of 5-min data")
    seq_len = CONFIG["seq_len"]
    X, y = prepare_dataset(df, seq_len)
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"])

    # Long model
    model_long = AttentionLSTMModel(
        input_dim=CONFIG["input_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        output_dim=2,
        dropout=CONFIG["dropout"],
        num_heads=CONFIG["num_heads"]
    ).to(device)
    print("Training LONG model (5-min)...")
    model_long = train_model(model_long, train_loader, val_loader, device, CONFIG)
    long_path = os.path.join(MODEL_DIR, "holy_grail_5m.pth")
    torch.save(model_long.state_dict(), long_path)
    print(f"Saved LONG model to {long_path}")

    # Short model
    model_short = AttentionLSTMModel(
        input_dim=CONFIG["input_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        output_dim=2,
        dropout=CONFIG["dropout"],
        num_heads=CONFIG["num_heads"]
    ).to(device)
    print("Training SHORT model (5-min)...")
    model_short = train_model(model_short, train_loader, val_loader, device, CONFIG)
    short_path = os.path.join(MODEL_DIR, "holy_grail_5m_short.pth")
    torch.save(model_short.state_dict(), short_path)
    print(f"Saved SHORT model to {short_path}")

    def evaluate(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total
    acc_long = evaluate(model_long, test_loader)
    acc_short = evaluate(model_short, test_loader)
    print(f"Test accuracy - LONG: {acc_long:.4f}, SHORT: {acc_short:.4f}")
