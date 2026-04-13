"""
Roughness index computation across all (exchange, asset, year, frequency) combos.

Reads processed realized-volatility parquet files from data/processed/,
applies the Cont-Das estimator (src/estimation/p_variation.py), and writes
summary tables to results/tables/.

Usage:
    python src/estimation/roughness.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.estimation.p_variation import estimate_roughness, k_opt, roughness_vs_K

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core per-series routine
# ---------------------------------------------------------------------------

def compute_roughness_for_series(
    rv: np.ndarray,
    cfg: dict,
    label: str = "",
) -> dict:
    """
    Run both standard and wide p-grid estimates on a single RV series.

    Returns a flat dict with all summary statistics.
    """
    N = len(rv)
    K = k_opt(N)

    std_cfg = cfg["estimation"]
    p_std = std_cfg["p_grid_standard"]
    p_wide = std_cfg["p_grid_wide"]
    n_steps = std_cfg["p_grid_steps"]

    result_std = estimate_roughness(
        rv, K=K, p_min=p_std[0], p_max=p_std[1], n_steps=n_steps
    )
    result_wide = estimate_roughness(
        rv, K=K, p_min=p_wide[0], p_max=p_wide[1], n_steps=n_steps
    )

    return {
        "label": label,
        "N": N,
        "K_opt": K,
        "n_opt": N // K,
        "L_used": K * (N // K),
        "H_standard": result_std["H"],
        "log_W_min_standard": result_std["log_W_min"],
        "log_W_max_standard": result_std["log_W_max"],
        "H_wide": result_wide["H"],
        "log_W_min_wide": result_wide["log_W_min"],
        "log_W_max_wide": result_wide["log_W_max"],
        "_curve_standard": result_std["curve"],
        "_curve_wide": result_wide["curve"],
    }


# ---------------------------------------------------------------------------
# K-stability diagnostic curves (paper Fig. 1)
# ---------------------------------------------------------------------------

def compute_K_stability(
    rv: np.ndarray,
    k_min: int = 50,
    k_max_factor: float = 0.5,
    n_steps: int = 50,
    p_min: float = 0.1,
    p_max: float = 4.0,
    p_steps: int = 100,
) -> np.ndarray:
    """
    Return H_hat(K) array for diagnostic plot.

    K is swept from k_min to min(k_max_factor * N, N-1) on a log scale.
    """
    N = len(rv)
    k_max = max(k_min + 1, int(k_max_factor * N))
    K_values = np.unique(
        np.round(np.logspace(np.log10(k_min), np.log10(k_max), n_steps)).astype(int)
    )
    return roughness_vs_K(rv, K_values, p_min=p_min, p_max=p_max, n_steps=p_steps)


# ---------------------------------------------------------------------------
# Pipeline: iterate over processed files
# ---------------------------------------------------------------------------

def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path("data/processed")
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    rv_files = sorted(processed_dir.glob("rv_*.parquet"))
    if not rv_files:
        log.warning("No processed RV files found in %s", processed_dir)
        return

    for fpath in rv_files:
        # Expected naming: rv_{exchange}_{asset}_{freq}_{year}.parquet
        stem = fpath.stem
        parts = stem.split("_")
        if len(parts) < 5:
            log.warning("Skipping unexpected filename: %s", fpath)
            continue

        # parts: ['rv', exchange, asset, freq, year]
        exchange = parts[1]
        asset = parts[2]
        freq = parts[3]
        year = parts[4]
        label = f"{exchange}/{asset}/{freq}/{year}"

        log.info("Processing %s", label)

        df = pd.read_parquet(fpath)
        if "rv" not in df.columns:
            log.warning("Column 'rv' not found in %s; skipping", fpath)
            continue

        rv = df["rv"].dropna().to_numpy(dtype=np.float64)
        if len(rv) < 100:
            log.warning("Too few observations (%d) in %s; skipping", len(rv), label)
            continue

        result = compute_roughness_for_series(rv, cfg, label=label)

        rows.append(
            {
                "exchange": exchange,
                "asset": asset,
                "frequency": freq,
                "year": year,
                "N": result["N"],
                "K_opt": result["K_opt"],
                "n_opt": result["n_opt"],
                "L_used": result["L_used"],
                "H_standard": result["H_standard"],
                "log_W_min_std": result["log_W_min_standard"],
                "log_W_max_std": result["log_W_max_standard"],
                "H_wide": result["H_wide"],
                "log_W_min_wide": result["log_W_min_wide"],
                "log_W_max_wide": result["log_W_max_wide"],
            }
        )

    summary = pd.DataFrame(rows)
    out_path = tables_dir / "p_variation_summary.csv"
    summary.to_csv(out_path, index=False)
    log.info("Saved roughness summary to %s", out_path)
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Roughness estimation pipeline")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
