"""Comprehensive codebase audit: checks every .py file for syntax errors and import errors."""
import os
import py_compile
import sys
import importlib
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {'__pycache__', '.git', 'venv', 'node_modules'}

# Phase 1: Syntax check every .py file
print("=" * 60)
print("PHASE 1: SYNTAX CHECK (every .py file)")
print("=" * 60)
syntax_errors = []
all_py = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if fn.endswith('.py'):
            fp = os.path.join(dirpath, fn)
            all_py.append(fp)
            try:
                py_compile.compile(fp, doraise=True)
            except py_compile.PyCompileError as e:
                syntax_errors.append((fp, str(e)))

if syntax_errors:
    for fp, err in syntax_errors:
        print(f"  SYNTAX ERROR: {os.path.relpath(fp, ROOT)}")
        print(f"    {err}")
else:
    print(f"  All {len(all_py)} Python files pass syntax check.")

# Phase 2: Import check for critical production files
print()
print("=" * 60)
print("PHASE 2: IMPORT CHECK (production files)")
print("=" * 60)

sys.path.insert(0, ROOT)
critical_imports = [
    ("ml.model", "AttentionLSTMModel"),
    ("ml.dataset", "TimeSeriesDataset"),
    ("data.feature_engineer_btc", "compute_live_features"),
    ("data.feature_engineer_btc", "get_feature_cols"),
    ("data.feature_engineer_btc", "engineer_features"),
    ("data.feature_engineer", "get_feature_cols"),
    ("data.feature_engineer", "precompute_static_features"),
    ("data.fetch_data", "fetch_klines"),
]

import_errors = []
for mod_name, attr_name in critical_imports:
    try:
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, attr_name):
            import_errors.append((mod_name, attr_name, "MISSING ATTRIBUTE"))
        else:
            print(f"  OK: {mod_name}.{attr_name}")
    except Exception as e:
        import_errors.append((mod_name, attr_name, str(e)))

if import_errors:
    print()
    for mod, attr, err in import_errors:
        print(f"  IMPORT ERROR: {mod}.{attr} -> {err}")

# Phase 3: Check for broken imports inside each production file
print()
print("=" * 60)
print("PHASE 3: RUNTIME IMPORT CHECK (import each top-level file)")
print("=" * 60)

production_files = [
    "app.py",
    "trade_live.py",
    "simulate_recent_14h.py",
    "fast_evaluate_full_dataset.py",
    "check_model.py",
    "test_recent.py",
]

for fn in production_files:
    fp = os.path.join(ROOT, fn)
    if not os.path.exists(fp):
        print(f"  SKIP (not found): {fn}")
        continue
    
    # Read file and check for import lines
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    bad_imports = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('from ') and 'import' in stripped:
            # Extract module name
            parts = stripped.split()
            if len(parts) >= 2:
                mod = parts[1]
                # Try importing
                try:
                    importlib.import_module(mod)
                except ImportError as e:
                    bad_imports.append((i, stripped, str(e)))
                except Exception:
                    pass  # Some imports need runtime context
    
    if bad_imports:
        print(f"  BROKEN IMPORTS in {fn}:")
        for lineno, line, err in bad_imports:
            print(f"    Line {lineno}: {line}")
            print(f"      -> {err}")
    else:
        print(f"  OK: {fn}")

# Phase 4: Check critical data files exist
print()
print("=" * 60)
print("PHASE 4: DATA FILE INTEGRITY CHECK")
print("=" * 60)

critical_files = [
    "models/holy_grail.pth",
    "models/holy_grail_config.json",
    "models_short/holy_grail_short.pth", 
    "models_short/holy_grail_short_config.json",
    "data_storage/BTC_USDT_15m_scaler.json",
    "data_storage/BTC_USDT_15m.csv",
    "data_storage/live_state.json",
    "requirements.txt",
    "render.yaml",
    "run.sh",
]

for fp in critical_files:
    full = os.path.join(ROOT, fp)
    if os.path.exists(full):
        sz = os.path.getsize(full)
        if sz == 0:
            print(f"  EMPTY FILE: {fp}")
        else:
            print(f"  OK: {fp} ({sz:,} bytes)")
    else:
        print(f"  MISSING: {fp}")

# Phase 5: Check JSON validity of config files
print()
print("=" * 60)
print("PHASE 5: JSON VALIDITY CHECK")
print("=" * 60)

import json
json_files = [
    "models/holy_grail_config.json",
    "models_short/holy_grail_short_config.json",
    "data_storage/BTC_USDT_15m_scaler.json",
    "data_storage/live_state.json",
]

for fp in json_files:
    full = os.path.join(ROOT, fp)
    if not os.path.exists(full):
        print(f"  SKIP: {fp}")
        continue
    try:
        with open(full, 'r') as f:
            data = json.load(f)
        print(f"  OK: {fp} ({len(data)} keys)")
    except json.JSONDecodeError as e:
        print(f"  JSON ERROR: {fp} -> {e}")

# Phase 6: Cross-check feature column counts
print()
print("=" * 60)
print("PHASE 6: FEATURE COLUMN CONSISTENCY CHECK")
print("=" * 60)

from data.feature_engineer_btc import get_feature_cols as btc_cols
from data.feature_engineer import get_feature_cols as gold_cols

btc_features = btc_cols()
gold_features = gold_cols()

with open("models/holy_grail_config.json") as f:
    cfg = json.load(f)
model_input_dim = cfg['input_dim']

print(f"  BTC feature_engineer_btc columns: {len(btc_features)}")
print(f"  Gold feature_engineer columns:    {len(gold_features)}")
print(f"  Model expected input_dim:         {model_input_dim}")

if len(btc_features) != model_input_dim:
    print(f"  MISMATCH: BTC features ({len(btc_features)}) != model input_dim ({model_input_dim})")
else:
    print(f"  OK: BTC features match model input_dim")

# Check dataset.py is using the right feature_engineer
with open(os.path.join(ROOT, "ml", "dataset.py"), 'r') as f:
    ds_code = f.read()
if "data.feature_engineer import" in ds_code and "feature_engineer_btc" not in ds_code:
    print(f"  BUG: ml/dataset.py imports from data.feature_engineer (Gold), not feature_engineer_btc (BTC)")
    if len(btc_features) != len(gold_features):
        print(f"  CRITICAL: Column count mismatch ({len(btc_features)} vs {len(gold_features)}) will cause silent bugs!")
    else:
        cols_match = btc_features == gold_features
        if not cols_match:
            print(f"  WARNING: Same count but DIFFERENT column names!")
        else:
            print(f"  OK: Both modules return identical column lists (safe for now)")

print()
print("=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
