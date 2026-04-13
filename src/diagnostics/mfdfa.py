"""
Multifractal Detrended Fluctuation Analysis (MF-DFA).

Implements the Kantelhardt et al. (2002) algorithm and the shuffle-based
robustness control described in Pontiggia (2025) §3.6.2.

Algorithm (Kantelhardt et al. 2002):

  1. Compute the profile:  Y(i) = Σ_{k=1}^i [X_k - <X>],  i = 1,...,N

  2. For each scale s:
       a. Divide Y into N_s = floor(N/s) non-overlapping segments of length s.
          Repeat from the end (2*N_s segments total).
       b. For each segment ν, fit a polynomial of order m (default m=1, DFA-1)
          and compute the residual variance:
              F²(ν, s) = (1/s) Σ_{i=1}^s [Y(ν) - ŷ_ν(i)]²
       c. q-th order fluctuation function:
              F_q(s) = { (1/(2*N_s)) Σ_{ν=1}^{2*N_s} [F²(ν,s)]^{q/2} }^{1/q}
          Special case q = 0 (L'Hôpital):
              F_0(s) = exp{ (1/(4*N_s)) Σ_ν ln[F²(ν,s)] }

  3. Scaling: F_q(s) ~ s^{H(q)}
     H(q) = slope of OLS regression  log F_q(s)  on  log s
     over intermediate scales.

Shuffle control (paper §3.6.2):
  Randomly permute the time indices of the input series n_shuffles times.
  Average H(q) over replicates.  Comparing original H(q) to shuffled H(q)
  isolates temporal (dynamic) from distributional multifractality.

Usage:
    python src/diagnostics/mfdfa.py --config config.yaml
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
# Core MF-DFA
# ---------------------------------------------------------------------------

def _profile(x: np.ndarray) -> np.ndarray:
    """Profile Y(i) = Σ_{k=1}^i [x_k - mean(x)]."""
    return np.cumsum(x - x.mean())


def _segment_variance(segment: np.ndarray, poly_order: int) -> float:
    """
    Detrend a segment with a polynomial of degree poly_order and return F²(ν,s).
    """
    s = len(segment)
    t = np.arange(1, s + 1, dtype=np.float64)
    coeffs = np.polyfit(t, segment, poly_order)
    trend = np.polyval(coeffs, t)
    return float(np.mean((segment - trend) ** 2))


def fluctuation_function(
    x: np.ndarray,
    scales: np.ndarray,
    q_values: np.ndarray,
    poly_order: int = 1,
) -> np.ndarray:
    """
    Compute F_q(s) for all (q, s) combinations.

    Parameters
    ----------
    x:
        Input time series, shape (N,).
    scales:
        Array of segment sizes s (integers).
    q_values:
        Array of moment orders q (can include 0, negatives).
    poly_order:
        Polynomial order for local detrending (paper uses 1 = DFA-1).

    Returns
    -------
    NDArray of shape (len(q_values), len(scales))
        F_q(s) values.
    """
    Y = _profile(x)
    N = len(Y)
    Fqs = np.full((len(q_values), len(scales)), np.nan)

    for s_idx, s in enumerate(scales):
        s = int(s)
        if s < poly_order + 2 or s > N // 2:
            continue

        N_s = N // s
        if N_s < 1:
            continue

        # Forward and backward segments (2*N_s total)
        F2_all = []
        for direction in (1, -1):
            seg_Y = Y if direction == 1 else Y[::-1]
            for v in range(N_s):
                seg = seg_Y[v * s: (v + 1) * s]
                f2 = _segment_variance(seg, poly_order)
                if f2 > 0:
                    F2_all.append(f2)

        if not F2_all:
            continue

        F2 = np.array(F2_all, dtype=np.float64)

        for q_idx, q in enumerate(q_values):
            if abs(q) < 1e-10:  # q ≈ 0: use log formula
                Fqs[q_idx, s_idx] = np.exp(0.5 * np.mean(np.log(F2)))
            else:
                Fqs[q_idx, s_idx] = np.mean(F2 ** (q / 2.0)) ** (1.0 / q)

    return Fqs


def hurst_exponents(
    x: np.ndarray,
    q_values: np.ndarray,
    min_scale: int = 10,
    max_scale_pct: float = 0.25,
    poly_order: int = 1,
    n_scales: int = 20,
) -> dict:
    """
    Estimate H(q) for all q via log-log regression of F_q(s) on s.

    Returns
    -------
    dict with keys: q_values, H, R2, scales, Fq (the full F_q(s) matrix)
    """
    N = len(x)
    max_scale = max(min_scale + 1, int(max_scale_pct * N))
    scales = np.unique(
        np.round(np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales))
        .astype(int)
    )
    scales = scales[scales >= poly_order + 2]
    scales = scales[scales <= N // 2]

    Fqs = fluctuation_function(x, scales, q_values, poly_order=poly_order)

    H = np.full(len(q_values), np.nan)
    R2 = np.full(len(q_values), np.nan)

    log_s = np.log(scales.astype(float))

    for q_idx in range(len(q_values)):
        fq = Fqs[q_idx, :]
        valid = np.isfinite(fq) & (fq > 0)
        if valid.sum() < 3:
            continue
        log_fq = np.log(fq[valid])
        ls = log_s[valid]
        # OLS: log F_q = H * log s + const
        A = np.column_stack([ls, np.ones(len(ls))])
        result = np.linalg.lstsq(A, log_fq, rcond=None)
        coeffs = result[0]
        H[q_idx] = coeffs[0]
        # R²
        fitted = A @ coeffs
        ss_res = np.sum((log_fq - fitted) ** 2)
        ss_tot = np.sum((log_fq - log_fq.mean()) ** 2)
        R2[q_idx] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "q_values": q_values,
        "H": H,
        "R2": R2,
        "scales": scales,
        "Fq": Fqs,
    }


# ---------------------------------------------------------------------------
# Shuffle control (paper §3.6.2)
# ---------------------------------------------------------------------------

def shuffle_control(
    x: np.ndarray,
    q_values: np.ndarray,
    n_shuffles: int = 20,
    **hurst_kwargs,
) -> dict:
    """
    Compute mean H(q) over n_shuffles random permutations of x.

    Permuting removes temporal correlations while preserving the marginal
    distribution, isolating dynamic from distributional multifractality.

    Returns
    -------
    dict with keys: H_mean (shape len(q)), H_std, H_all (n_shuffles × len(q))
    """
    H_all = []
    rng = np.random.default_rng(seed=42)
    for _ in range(n_shuffles):
        x_shuf = rng.permutation(x)
        result = hurst_exponents(x_shuf, q_values, **hurst_kwargs)
        H_all.append(result["H"])

    H_all = np.array(H_all)
    return {
        "H_mean": np.nanmean(H_all, axis=0),
        "H_std": np.nanstd(H_all, axis=0),
        "H_all": H_all,
    }


# ---------------------------------------------------------------------------
# Summary statistics for one series (paper Appendix Tables A.6, A.9)
# ---------------------------------------------------------------------------

def mfdfa_summary(
    x: np.ndarray,
    cfg: dict,
    compute_shuffle: bool = True,
) -> dict:
    """
    Run MF-DFA on x and optionally on shuffled versions.

    Returns a dict suitable for appending to a results table.
    """
    mf_cfg = cfg["mfdfa"]
    q_values = np.linspace(mf_cfg["q_min"], mf_cfg["q_max"], mf_cfg["q_steps"])

    hurst_kwargs = dict(
        min_scale=mf_cfg["min_scale"],
        max_scale_pct=mf_cfg["max_scale_pct"],
        poly_order=mf_cfg["poly_order"],
    )

    orig = hurst_exponents(x, q_values, **hurst_kwargs)
    H = orig["H"]

    result = {
        "H_min": float(np.nanmin(H)),
        "H_max": float(np.nanmax(H)),
        "H_mean": float(np.nanmean(H)),
        "spectral_width": float(np.nanmax(H) - np.nanmin(H)),
        "H_q": H,
        "q_values": q_values,
        "R2": orig["R2"],
    }

    if compute_shuffle:
        shuf = shuffle_control(
            x, q_values, n_shuffles=mf_cfg["n_shuffles"], **hurst_kwargs
        )
        result.update(
            {
                "H_shuffled_mean": shuf["H_mean"],
                "H_shuffled_std": shuf["H_std"],
                "spectral_width_shuffled": float(
                    np.nanmax(shuf["H_mean"]) - np.nanmin(shuf["H_mean"])
                ),
            }
        )

    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path("data/processed")
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    rv_files = sorted(processed_dir.glob("rv_*.parquet"))
    for fpath in rv_files:
        parts = fpath.stem.split("_")
        if len(parts) < 5:
            continue
        exchange, asset, freq, year = parts[1], parts[2], parts[3], parts[4]
        label = f"{exchange}/{asset}/{freq}/{year}"

        df = pd.read_parquet(fpath)
        if "rv" not in df.columns:
            continue
        rv = df["rv"].dropna().to_numpy(dtype=np.float64)
        if len(rv) < 100:
            continue

        log.info("MF-DFA: %s", label)
        summary = mfdfa_summary(rv, cfg, compute_shuffle=True)

        rows.append(
            {
                "exchange": exchange,
                "asset": asset,
                "frequency": freq,
                "year": year,
                "H_min": summary["H_min"],
                "H_max": summary["H_max"],
                "H_mean": summary["H_mean"],
                "spectral_width": summary["spectral_width"],
                "spectral_width_shuffled": summary.get("spectral_width_shuffled"),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = tables_dir / "mfdfa_summary.csv"
    out_df.to_csv(out_path, index=False)
    log.info("MF-DFA summary → %s", out_path)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="MF-DFA diagnostics")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
