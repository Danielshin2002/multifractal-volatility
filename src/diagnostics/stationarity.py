"""
Stationarity diagnostics (paper §3.6.1).

Three complementary tests applied to each RV series:

1. Augmented Dickey-Fuller (ADF) unit-root test — lag order by AIC.
       ΔRV_t = α RV_{t-1} + Σ β_i ΔRV_{t-i} + ε_t

2. Rolling mean and variance stability:
       std( mean_{t-w:t}(RV) ),   std( var_{t-w:t}(RV) )
   rolling window w = max(10, floor(0.05 * N)).

3. Structural break detection via binary segmentation with L²-cost,
   maximum 5 breakpoints.

Usage:
    python src/diagnostics/stationarity.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
import yaml
from statsmodels.tsa.stattools import adfuller

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ADF test (paper §3.6.1)
# ---------------------------------------------------------------------------

def run_adf(rv: np.ndarray, significance: float = 0.05) -> dict:
    """
    ADF test with AIC lag selection.

    Returns dict: {adf_stat, p_value, lags_used, critical_values, is_stationary}.
    """
    result = adfuller(rv, autolag="AIC", regression="c")
    adf_stat, p_value, lags_used, *_ = result
    crit_vals = result[4]
    return {
        "adf_stat": float(adf_stat),
        "p_value": float(p_value),
        "lags_used": int(lags_used),
        "cv_1pct": float(crit_vals["1%"]),
        "cv_5pct": float(crit_vals["5%"]),
        "cv_10pct": float(crit_vals["10%"]),
        "is_stationary": p_value < significance,
    }


# ---------------------------------------------------------------------------
# Rolling stability (paper §3.6.1)
# ---------------------------------------------------------------------------

def rolling_stability(
    rv: np.ndarray,
    window_pct: float = 0.05,
    min_window: int = 10,
) -> dict:
    """
    Compute std of rolling mean and rolling variance.

    window w = max(min_window, floor(window_pct * N)).
    """
    N = len(rv)
    w = max(min_window, int(np.floor(window_pct * N)))
    series = pd.Series(rv)
    roll_mean = series.rolling(w).mean().dropna()
    roll_var = series.rolling(w).var().dropna()
    return {
        "window_size": w,
        "std_rolling_mean": float(roll_mean.std()),
        "std_rolling_var": float(roll_var.std()),
        "mean_of_means": float(roll_mean.mean()),
        "mean_of_vars": float(roll_var.mean()),
    }


# ---------------------------------------------------------------------------
# Structural break detection (paper §3.6.1)
# ---------------------------------------------------------------------------

def detect_breaks(rv: np.ndarray, max_breaks: int = 5) -> dict:
    """
    Binary segmentation with L²-cost; up to max_breaks breakpoints.

    Returns dict: {n_breaks, breakpoint_indices, breakpoint_positions}.
    """
    algo = rpt.Binseg(model="l2").fit(rv)
    # predict returns breakpoint indices (1-based, includes N at end)
    bkpts = algo.predict(n_bkps=max_breaks)
    # Exclude the last element (= N) which is always returned
    bkpt_indices = [b - 1 for b in bkpts[:-1]]
    return {
        "n_breaks": len(bkpt_indices),
        "breakpoint_indices": bkpt_indices,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path("data/processed")
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    sig = cfg["stationarity"]["adf_significance"]
    roll_pct = cfg["stationarity"]["rolling_window_pct"]
    max_bkpts = cfg["stationarity"]["max_breakpoints"]

    adf_rows, rolling_rows, break_rows = [], [], []

    rv_files = sorted(processed_dir.glob("rv_*.parquet"))
    for fpath in rv_files:
        stem = fpath.stem
        parts = stem.split("_")
        if len(parts) < 5:
            continue
        exchange, asset, freq, year = parts[1], parts[2], parts[3], parts[4]
        label = f"{exchange}/{asset}/{freq}/{year}"

        df = pd.read_parquet(fpath)
        if "rv" not in df.columns:
            continue
        rv = df["rv"].dropna().to_numpy(dtype=np.float64)
        if len(rv) < 50:
            continue

        log.info("Stationarity diagnostics: %s", label)

        # ADF
        adf = run_adf(rv, significance=sig)
        adf_rows.append({"label": label, "exchange": exchange, "asset": asset,
                         "frequency": freq, "year": year, **adf})

        # Rolling stability
        rs = rolling_stability(rv, window_pct=roll_pct)
        rolling_rows.append({"label": label, "exchange": exchange, "asset": asset,
                              "frequency": freq, "year": year, **rs})

        # Structural breaks
        sb = detect_breaks(rv, max_breaks=max_bkpts)
        break_rows.append({"label": label, "exchange": exchange, "asset": asset,
                           "frequency": freq, "year": year, **sb})

    pd.DataFrame(adf_rows).to_csv(tables_dir / "adf_results.csv", index=False)
    pd.DataFrame(rolling_rows).to_csv(tables_dir / "rolling_stability.csv", index=False)
    pd.DataFrame(break_rows).to_csv(tables_dir / "structural_breaks.csv", index=False)
    log.info("Stationarity tables written to %s", tables_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Stationarity diagnostics")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
