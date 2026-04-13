#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-config.yaml}

echo "=== Step 1: Fetch OHLCV data ==="
python src/data/fetch.py --config "$CONFIG"

echo "=== Step 2: Preprocess (90-day window selection + completeness filter) ==="
python src/data/preprocess.py --config "$CONFIG"

echo "=== Step 3: Construct realized volatility ==="
python src/data/realized_vol.py --config "$CONFIG"

echo "=== Step 4: Roughness estimation (p-variation) ==="
python src/estimation/roughness.py --config "$CONFIG"

echo "=== Step 5: Stationarity diagnostics ==="
python src/diagnostics/stationarity.py --config "$CONFIG"

echo "=== Step 6a: MF-DFA ==="
python src/diagnostics/mfdfa.py --config "$CONFIG"

echo "=== Step 6b: Moment scaling ==="
python src/diagnostics/moment_scaling.py --config "$CONFIG"

echo "=== Step 6c: Wavelet leaders ==="
python src/diagnostics/wavelet_leaders.py --config "$CONFIG"

echo "=== Step 7: Generate figures ==="
python src/visualization/p_variation_plots.py --config "$CONFIG"
python src/visualization/multifractal_plots.py --config "$CONFIG"
python src/visualization/comparison_plots.py --config "$CONFIG"

echo "=== Pipeline complete. Results in results/ ==="
