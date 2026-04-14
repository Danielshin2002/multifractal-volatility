"""
Preprocessing: 90-day best-window selection and completeness filtering.

Implements exactly the procedure from Pontiggia (2025) §3.2:

  For each calendar year Y, select the most complete 90-day consecutive window
  of 1-minute observations.  T = 129,600 minutes; the search space sweeps in
  daily steps (1,440 minutes); ties broken by choosing the most recent window.

  For higher frequencies (5m, 10m, 15m), resample the selected 1-minute window
  using only complete intervals (N_f valid 1-minute observations per bar).

  Discard any (year, frequency) combination where completeness < 90%.

Usage:
    python src/data/preprocess.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

# Paper: T = 90 * 1440 = 129,600 minutes
WINDOW_MINUTES = 129_600
STEP_MINUTES = 1_440  # daily increment


def select_best_window(
    df_1m: pd.DataFrame,
    year: int,
    window_minutes: int = WINDOW_MINUTES,
    step_minutes: int = STEP_MINUTES,
) -> pd.DataFrame | None:
    """
    Find the most complete consecutive 90-day window within `year`.

    Paper algorithm:
        - Search all windows of length T starting at ts, stepping by 1 day
        - Count missing Close observations m(ts) in each window
        - (t*_s, t*_e) = argmin_ts m(ts); on ties pick the most recent window

    Returns
    -------
    Trimmed 1-minute DataFrame for the best window, or None if no
    candidate window exists.
    """
    year_mask = df_1m.index.year == year
    df_year = df_1m[year_mask].copy()

    if df_year.empty:
        log.warning("No data for year %d", year)
        return None

    year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    year_end = pd.Timestamp(f"{year}-12-31 23:59:00", tz="UTC")

    # Build a complete 1-minute index for the year
    full_1m_idx = pd.date_range(year_start, year_end, freq="1min", tz="UTC")
    df_full = df_year.reindex(full_1m_idx)  # NaN where missing

    n_full = len(full_1m_idx)
    if n_full < window_minutes:
        log.warning("Year %d has fewer than %d minutes; skipping", year, window_minutes)
        return None

    # Sweep windows
    best_missing = np.inf
    best_start_idx = -1

    n_windows = (n_full - window_minutes) // step_minutes + 1
    for i in range(n_windows):
        start_i = i * step_minutes
        end_i = start_i + window_minutes
        window_close = df_full["close"].iloc[start_i:end_i]
        missing = window_close.isna().sum()
        # <= to prefer most recent window on ties (iterate forward → last tie wins)
        if missing <= best_missing:
            best_missing = missing
            best_start_idx = start_i

    if best_start_idx < 0:
        log.warning("No valid window found for year %d", year)
        return None

    best_slice = df_full.iloc[best_start_idx: best_start_idx + window_minutes]
    completeness = 1.0 - best_missing / window_minutes
    log.info(
        "Year %d: best window starts %s, completeness=%.1f%%",
        year,
        best_slice.index[0].date(),
        completeness * 100,
    )
    return best_slice


def resample_to_freq(
    df_1m: pd.DataFrame,
    freq_minutes: int,
) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV to freq_minutes.

    Only retain bars where all N_f = freq_minutes 1-minute intervals are
    present (i.e. complete intervals only, per paper §3.2).
    """
    rule = f"{freq_minutes}min"

    # Count valid 1-minute observations per interval
    df_1m = df_1m.copy()
    df_1m["_valid"] = (~df_1m["close"].isna()).astype(int)

    agg = df_1m.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        _n_valid=("_valid", "sum"),
    )

    # Keep only fully populated intervals
    complete = agg[agg["_n_valid"] == freq_minutes].drop(columns=["_n_valid"])
    return complete.dropna(subset=["close"])


def compute_completeness(
    df: pd.DataFrame, freq_minutes: int, window_minutes: int = WINDOW_MINUTES
) -> float:
    n_expected = window_minutes // freq_minutes
    n_valid = len(df.dropna(subset=["close"]))
    return n_valid / n_expected


def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    tables_dir = Path("results/tables")
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    completeness_threshold = cfg["preprocessing"]["completeness_threshold"]
    window_minutes = cfg["preprocessing"]["window_minutes"]
    step_minutes = cfg["preprocessing"]["step_minutes"]
    frequencies = cfg["frequencies"]
    date_range = cfg["date_range"]
    years = list(range(int(date_range["start"][:4]), int(date_range["end"][:4]) + 1))

    freq_minute_map = {"1m": 1, "5m": 5, "10m": 10, "15m": 15}
    completeness_rows = []

    for exchange in cfg["exchanges"]:
        for asset in cfg["assets"]:
            asset_slug = asset.replace("/", "_")
            raw_1m_path = raw_dir / exchange / asset_slug / "1m.parquet"

            if not raw_1m_path.exists():
                log.warning("Missing raw 1m data: %s", raw_1m_path)
                continue

            log.info("Loading %s", raw_1m_path)
            df_1m = pd.read_parquet(raw_1m_path)

            for year in years:
                # Select best 90-day window at 1-minute resolution
                window_1m = select_best_window(
                    df_1m, year,
                    window_minutes=window_minutes,
                    step_minutes=step_minutes,
                )
                if window_1m is None:
                    continue

                for freq in frequencies:
                    freq_min = freq_minute_map[freq]

                    if freq_min == 1:
                        df_freq = window_1m.dropna(subset=["close"])
                    else:
                        df_freq = resample_to_freq(window_1m, freq_min)

                    comp = compute_completeness(df_freq, freq_min, window_minutes)
                    completeness_rows.append(
                        {
                            "exchange": exchange,
                            "asset": asset,
                            "frequency": freq,
                            "year": year,
                            "valid_bars": len(df_freq),
                            "expected_bars": window_minutes // freq_min,
                            "completeness_pct": round(comp * 100, 2),
                        }
                    )

                    if comp < completeness_threshold:
                        log.info(
                            "Skipping %s/%s/%s/%d: completeness=%.1f%% < %.0f%%",
                            exchange, asset, freq, year, comp * 100,
                            completeness_threshold * 100,
                        )
                        continue

                    out_path = (
                        processed_dir
                        / f"ohlcv_{exchange}_{asset_slug}_{freq}_{year}.parquet"
                    )
                    df_freq.to_parquet(out_path, compression="gzip")
                    log.info("Saved %s (%d bars)", out_path, len(df_freq))

    # Write completeness table
    comp_df = pd.DataFrame(completeness_rows)
    comp_path = tables_dir / "completeness.csv"
    comp_df.to_csv(comp_path, index=False)
    log.info("Completeness table → %s", comp_path)
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Preprocess OHLCV data")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
