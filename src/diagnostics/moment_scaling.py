"""
Log-log moment scaling analysis (paper §3.6.2, eq. 4).

Estimates empirical scaling exponents ζ_q via:

    M_q(τ) = E[ |X_{t+τ} − X_t|^q ]  ~  τ^{ζ_q}

For each moment order q ∈ [−4, 4], ζ_q is the OLS slope of
    log M_q(τ)  on  log τ
over a grid of lags τ.

Multifractality diagnostic: non-linearity of q ↦ ζ_q.
Reference: Pontiggia (2025) §3.6.2, Bacry et al. (2001).

Usage:
    python src/diagnostics/moment_scaling.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)


def structure_function(
    x: np.ndarray,
    tau: int,
    q: float,
) -> float:
    """
    Empirical q-th order structure function at lag τ:
        M_q(τ) = mean( |x_{t+τ} − x_t|^q )

    For q < 0 and increments close to zero, a small epsilon guards against
    division by zero.
    """
    increments = np.abs(x[tau:] - x[:-tau])
    if q < 0:
        # guard: replace zeros with machine epsilon
        increments = np.where(increments == 0, np.finfo(float).eps, increments)
    with np.errstate(invalid="ignore", over="ignore"):
        return float(np.mean(increments ** q))


def scaling_exponents(
    x: np.ndarray,
    q_values: np.ndarray,
    tau_min: int = 1,
    tau_max_pct: float = 0.10,
    n_taus: int = 30,
) -> dict:
    """
    Estimate ζ_q for each q by regressing log M_q(τ) on log τ.

    Parameters
    ----------
    x:
        Input time series (e.g. realized volatility).
    q_values:
        Moment orders.
    tau_min:
        Minimum lag.
    tau_max_pct:
        Maximum lag as fraction of series length.
    n_taus:
        Number of lags on a log scale.

    Returns
    -------
    dict with keys: q_values, zeta_q, R2, taus, Mq (matrix n_q × n_tau)
    """
    N = len(x)
    tau_max = max(tau_min + 1, int(tau_max_pct * N))
    taus = np.unique(
        np.round(np.logspace(np.log10(tau_min), np.log10(tau_max), n_taus)).astype(int)
    )
    taus = taus[taus < N]

    Mq = np.full((len(q_values), len(taus)), np.nan)
    for q_idx, q in enumerate(q_values):
        for t_idx, tau in enumerate(taus):
            val = structure_function(x, int(tau), q)
            if np.isfinite(val) and val > 0:
                Mq[q_idx, t_idx] = val

    zeta_q = np.full(len(q_values), np.nan)
    R2 = np.full(len(q_values), np.nan)
    log_tau = np.log(taus.astype(float))

    for q_idx in range(len(q_values)):
        mq = Mq[q_idx, :]
        valid = np.isfinite(mq) & (mq > 0)
        if valid.sum() < 3:
            continue
        log_mq = np.log(mq[valid])
        lt = log_tau[valid]
        A = np.column_stack([lt, np.ones(len(lt))])
        res = np.linalg.lstsq(A, log_mq, rcond=None)
        coeffs = res[0]
        zeta_q[q_idx] = coeffs[0]
        fitted = A @ coeffs
        ss_res = np.sum((log_mq - fitted) ** 2)
        ss_tot = np.sum((log_mq - log_mq.mean()) ** 2)
        R2[q_idx] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "q_values": q_values,
        "zeta_q": zeta_q,
        "R2": R2,
        "taus": taus,
        "Mq": Mq,
    }


def spectral_width(zeta_q: np.ndarray) -> float:
    finite = zeta_q[np.isfinite(zeta_q)]
    if len(finite) < 2:
        return np.nan
    return float(finite.max() - finite.min())


def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path("data/processed")
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    ms_cfg = cfg["moment_scaling"]
    q_values = np.linspace(ms_cfg["q_min"], ms_cfg["q_max"], ms_cfg["q_steps"])

    rows = []

    for fpath in sorted(processed_dir.glob("rv_*.parquet")):
        parts = fpath.stem.split("_")
        if len(parts) < 6:
            continue
        exchange, asset, freq, year = parts[1], f"{parts[2]}_{parts[3]}", parts[4], parts[5]
        label = f"{exchange}/{asset}/{freq}/{year}"

        df = pd.read_parquet(fpath)
        if "rv" not in df.columns:
            continue
        rv = df["rv"].dropna().to_numpy(dtype=np.float64)
        if len(rv) < 100:
            continue

        log.info("Moment scaling: %s", label)
        result = scaling_exponents(
            rv,
            q_values,
            tau_min=ms_cfg["tau_min"],
            tau_max_pct=ms_cfg["tau_max_pct"],
        )
        sw = spectral_width(result["zeta_q"])

        rows.append(
            {
                "exchange": exchange,
                "asset": asset,
                "frequency": freq,
                "year": year,
                "spectral_width_zeta": sw,
                "zeta_q_min": float(np.nanmin(result["zeta_q"])),
                "zeta_q_max": float(np.nanmax(result["zeta_q"])),
                "mean_R2": float(np.nanmean(result["R2"])),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = tables_dir / "moment_scaling_summary.csv"
    out_df.to_csv(out_path, index=False)
    log.info("Moment scaling summary → %s", out_path)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Log-log moment scaling analysis")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
