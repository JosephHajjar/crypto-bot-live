import os

def create_gold_scripts():
    path = r'c:\Users\asdf\.gemini\antigravity\scratch\ml_trading_bot'

    # LONG
    with open(os.path.join(path, "optuna_search.py"), "r", encoding="utf-8") as f:
        content_long = f.read()

    content_long = content_long.replace("SYMBOL = 'BTC/USDT'", "SYMBOL = 'PAXG/USDT'")
    content_long = content_long.replace("MODEL_DIR = 'models'", "MODEL_DIR = 'models_gold_long'")
    content_long = content_long.replace("STATE_FILE = 'optuna_state.json'", "STATE_FILE = 'optuna_gold_long_state.json'")
    content_long = content_long.replace("LOG_FILE = 'optuna_output.log'", "LOG_FILE = 'optuna_gold_long.log'")
    content_long = content_long.replace("n_trials=2000", "n_trials=50")
    content_long = content_long.replace("n_startup_trials=10", "n_startup_trials=5")
    content_long = content_long.replace("n_warmup_steps=5", "n_warmup_steps=3")
    content_long = content_long.replace("n_startup_trials=15", "n_startup_trials=8")
    content_long = content_long.replace("TAKE_PROFIT = 0.015", "TAKE_PROFIT = 0.0125")
    content_long = content_long.replace("STOP_LOSS = 0.0075", "STOP_LOSS = 0.006")

    with open(os.path.join(path, "train_gold_long.py"), "w", encoding="utf-8") as f:
        f.write(content_long)

    # SHORT
    with open(os.path.join(path, "optuna_search_short.py"), "r", encoding="utf-8") as f:
        content_short = f.read()

    content_short = content_short.replace("SYMBOL = 'BTC/USDT'", "SYMBOL = 'PAXG/USDT'")
    content_short = content_short.replace("MODEL_DIR = 'models_short'", "MODEL_DIR = 'models_gold_short'")
    content_short = content_short.replace("STATE_FILE = 'optuna_state_short.json'", "STATE_FILE = 'optuna_gold_short_state.json'")
    content_short = content_short.replace("LOG_FILE = 'optuna_output_short.log'", "LOG_FILE = 'optuna_gold_short.log'")
    content_short = content_short.replace("n_trials=2000", "n_trials=50")
    content_short = content_short.replace("n_startup_trials=10", "n_startup_trials=5")
    content_short = content_short.replace("n_warmup_steps=5", "n_warmup_steps=3")
    content_short = content_short.replace("n_startup_trials=15", "n_startup_trials=8")
    
    # modify tp and sl float ranges
    content_short = content_short.replace("trial.suggest_float('take_profit', 0.005, 0.04, step=0.0025)", "trial.suggest_float('take_profit', 0.004, 0.02, step=0.002)")
    content_short = content_short.replace("trial.suggest_float('stop_loss', 0.005, 0.04, step=0.0025)", "trial.suggest_float('stop_loss', 0.004, 0.02, step=0.002)")

    with open(os.path.join(path, "train_gold_short.py"), "w", encoding="utf-8") as f:
        f.write(content_short)

    print("Scripts created successfully.")

if __name__ == "__main__":
    create_gold_scripts()
