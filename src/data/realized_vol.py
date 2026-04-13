"""
Realized volatility construction (paper §3.3).

Two constructions from Pontiggia (2025):

1. Simple absolute-return proxy (paper eq. used throughout):
       RV_t = |r_t|
   where r_t = log(close_t / close_{t-1}) at the sampling frequency.

2. Noise-robust multi-scale variant (paper §3.3, Fukasawa et al. 2022):
       RV^Δ_t = sqrt( sum_{i=1}^{Δ} r²_{t,i} )
   where {r_{t,i}} are 1-minute log-returns within the Δ-minute interval.
   This mitigates white-noise contamination at low frequencies.

Output: Parquet files to data/processed/ named rv_{exchange}_{asset}_{freq}_{year}.parquet
        with columns [timestamp, rv, rv_robust] (rv_robust only for freq > 1m).

Usage:
    python src/data/realized_vol.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log-return computation
# ---------------------------------------------------------------------------

def log_returns(close: pd.Series) -> pd.Series:
    """Compute log(close_t / close_{t-1}) with NaN at t=0."""
    return np.log(close / close.shift(1))


# ---------------------------------------------------------------------------
# RV_t = |r_t|  (paper §3.3, primary construction)
# ---------------------------------------------------------------------------

def simple_rv(df: pd.DataFrame) -> pd.Series:
    """
    Absolute log-return series.

        RV_t = |log(close_t / close_{t-1})|

    First observation is NaN and dropped by caller.
    """
    r = log_returns(df["close"])
    return r.abs().rename("rv")


# ---------------------------------------------------------------------------
# Noise-robust RV^Δ_t  (paper §3.3 footnote, Fukasawa et al. 2022)
# ---------------------------------------------------------------------------

def robust_rv(
    df_1m: pd.DataFrame,
    freq_minutes: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.Series:
    """
    Noise-robust realised volatility at frequency freq_minutes.

        RV^Δ_t = sqrt( sum_{i=1}^{Δ} r²_{t,i} )

    where {r_{t,i}} are 1-minute log-returns inside the Δ-minute bar ending at t.

    Parameters
    ----------
    df_1m:
        1-minute OHLCV DataFrame indexed by UTC timestamps.
    freq_minutes:
        Target bar frequency in minutes.
    window_start, window_end:
        The 90-day analysis window.

    Returns
    -------
    pd.Series indexed by bar-end timestamps.
    """
    mask = (df_1m.index >= window_start) & (df_1m.index <= window_end)
    df_slice = df_1m[mask].copy()
    r_1m = log_returns(df_slice["close"]).dropna()

    # Resample r² to the target frequency (sum → sqrt)
    r2 = (r_1m ** 2).resample(f"{freq_minutes}min").agg(
        rv_sq=("rv_sq" if False else r_1m.name, "sum"),  # sum of squared 1m returns
        n_valid=(r_1m.name, "count"),
    )
    # Only keep fully populated bars
    r2 = r2[r2["n_valid"] == freq_minutes]
    result = np.sqrt(r2["rv_sq"]).rename("rv_robust")
    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    processed_dir.mkdir(parents=True, exist_ok=True)

    freq_minute_map = {"1m": 1, "5m": 5, "10m": 10, "15m": 15}
    date_range = cfg["date_range"]
    years = list(range(int(date_range["start"][:4]), int(date_range["end"][:4]) + 1))

    for exchange in cfg["exchanges"]:
        for asset in cfg["assets"]:
            asset_slug = asset.replace("/", "_")

            # Load 1-minute raw data for noise-robust variant
            raw_1m_path = raw_dir / exchange / asset_slug / "1m.parquet"
            df_1m: pd.DataFrame | None = None
            if raw_1m_path.exists():
                df_1m = pd.read_parquet(raw_1m_path)

            for freq in cfg["frequencies"]:
                freq_min = freq_minute_map[freq]

                for year in years:
                    ohlcv_path = (
                        processed_dir
                        / f"ohlcv_{exchange}_{asset_slug}_{freq}_{year}.parquet"
                    )
                    if not ohlcv_path.exists():
                        continue

                    df = pd.read_parquet(ohlcv_path)
                    if "close" not in df.columns or df.empty:
                        log.warning("No 'close' column in %s; skipping", ohlcv_path)
                        continue

                    # Primary: RV_t = |r_t|
                    rv = simple_rv(df).dropna()

                    # Noise-robust: RV^Δ_t (only meaningful for freq > 1m)
                    rv_df = rv.to_frame()
                    if freq_min > 1 and df_1m is not None:
                        try:
                            window_start = df.index[0]
                            window_end = df.index[-1]
                            rv_robust = robust_rv(df_1m, freq_min, window_start, window_end)
                            rv_df = rv_df.join(rv_robust, how="left")
                        except Exception as e:
                            log.warning(
                                "Robust RV failed for %s/%s/%s/%d: %s",
                                exchange, asset, freq, year, e,
                            )

                    out_path = (
                        processed_dir
                        / f"rv_{exchange}_{asset_slug}_{freq}_{year}.parquet"
                    )
                    rv_df.to_parquet(out_path, compression="gzip")
                    log.info(
                        "Saved %s (%d obs)", out_path, len(rv_df)
                    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Construct realized volatility series")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
