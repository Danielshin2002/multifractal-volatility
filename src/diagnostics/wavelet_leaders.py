"""
Wavelet leaders multifractal analysis (paper §3.6.2, eq. 6).

Estimates scaling exponents ζ_q via:

    E[ |d_{j,k}|^q ]  ~  2^{j ζ_q}

where d_{j,k} are wavelet leader coefficients at dyadic scale 2^j.

Wavelet leaders are the local suprema of wavelet coefficients:
    l_{j,k} = sup_{j' <= j, |k' * 2^{j'-j} - k| <= 1} |w_{j',k'}|

where w_{j',k'} are DWT detail coefficients at level j', position k'.

Scaling exponents ζ_q are obtained by regressing
    log2 M_q(2^j)  on  j
over the intermediate scale range [min_scale_j, max_scale_j].

Reference: Pontiggia (2025) §3.6.2; Wendt et al. (2007); Bacry et al. (2001).

Usage:
    python src/diagnostics/wavelet_leaders.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import yaml

log = logging.getLogger(__name__)


def compute_wavelet_leaders(
    x: np.ndarray,
    wavelet: str = "db4",
    max_level: int = 10,
) -> list[np.ndarray]:
    """
    Compute wavelet leader coefficients for all dyadic scales.

    Parameters
    ----------
    x:
        Input time series.
    wavelet:
        PyWavelets wavelet name (paper uses db4).
    max_level:
        Maximum decomposition level.

    Returns
    -------
    List of length max_level.
        leaders[j-1] contains |leader| coefficients at scale 2^j (1-indexed).
    """
    # DWT decomposition (detail coefficients at each level)
    actual_max = min(max_level, pywt.dwt_max_level(len(x), wavelet))
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=actual_max, mode="periodization")
    # coeffs[0] = approx; coeffs[1] = detail level 1 (finest), ...
    # We use detail levels only: coeffs[1], ..., coeffs[actual_max]
    detail_coeffs = [np.abs(c) for c in coeffs[1:]]  # index 0 = finest scale

    n_levels = len(detail_coeffs)
    leaders = []

    for j in range(n_levels):  # j = 0 means level 1 (scale 2^1)
        d_j = detail_coeffs[j]
        n_j = len(d_j)

        # Leader at (j, k): max over same-scale 3-neighbours and all finer scales
        # within the support: l_{j,k} = max_{j'<=j, k'∈λ_{j,k}} |w_{j',k'}|
        leader_j = np.zeros(n_j)
        for k in range(n_j):
            # Finer scale contributions: levels 0, ..., j
            max_val = 0.0
            for jp in range(j + 1):
                d_jp = detail_coeffs[jp]
                ratio = 2 ** (j - jp)  # how many fine-scale coeffs per coarse coeff
                start = k * ratio
                end = min((k + 1) * ratio, len(d_jp))
                if start < len(d_jp):
                    max_val = max(max_val, np.max(d_jp[start:end]))
            # Same-scale neighbour padding (3 adjacent coefficients)
            lo = max(0, k - 1)
            hi = min(n_j, k + 2)
            same_scale_max = np.max(detail_coeffs[j][lo:hi])
            leader_j[k] = max(max_val, same_scale_max)

        leaders.append(leader_j)

    return leaders


def wavelet_scaling_exponents(
    x: np.ndarray,
    q_values: np.ndarray,
    wavelet: str = "db4",
    min_j: int = 2,
    max_j: int = 8,
) -> dict:
    """
    Estimate ζ_q from wavelet leaders.

    For each q, regress  log2( (1/n_j) Σ_k |l_{j,k}|^q )  on  j.

    Parameters
    ----------
    x:
        Input series.
    q_values:
        Moment orders.
    wavelet:
        Wavelet filter name.
    min_j, max_j:
        Dyadic scale range for regression (1-indexed).
        Paper recommends skipping first 2 scales (microstructure noise)
        and capping at scale 8 (non-stationary drift).

    Returns
    -------
    dict with keys: q_values, zeta_q, R2, j_levels, log2_Mq (n_q × n_j matrix)
    """
    leaders = compute_wavelet_leaders(x, wavelet=wavelet, max_level=max_j)
    n_levels = len(leaders)

    # Build valid j range
    j_range = [j for j in range(min_j - 1, min(max_j, n_levels)) if len(leaders[j]) >= 4]
    if len(j_range) < 2:
        return {
            "q_values": q_values,
            "zeta_q": np.full(len(q_values), np.nan),
            "R2": np.full(len(q_values), np.nan),
            "j_levels": np.array(j_range) + 1,
            "log2_Mq": np.full((len(q_values), max(1, len(j_range))), np.nan),
        }

    log2_Mq = np.full((len(q_values), len(j_range)), np.nan)

    for j_idx, j in enumerate(j_range):
        lj = leaders[j]
        for q_idx, q in enumerate(q_values):
            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                if q < 0:
                    # Guard: replace zeros with machine epsilon
                    lj_safe = np.where(lj == 0, np.finfo(float).eps, lj)
                else:
                    lj_safe = lj
                moment = np.mean(lj_safe ** q)
            if np.isfinite(moment) and moment > 0:
                log2_Mq[q_idx, j_idx] = np.log2(moment)

    j_vals = np.array(j_range, dtype=float) + 1.0  # 1-indexed scale j

    zeta_q = np.full(len(q_values), np.nan)
    R2 = np.full(len(q_values), np.nan)

    for q_idx in range(len(q_values)):
        row = log2_Mq[q_idx, :]
        valid = np.isfinite(row)
        if valid.sum() < 3:
            continue
        jv = j_vals[valid]
        rv = row[valid]
        A = np.column_stack([jv, np.ones(len(jv))])
        res = np.linalg.lstsq(A, rv, rcond=None)
        coeffs = res[0]
        zeta_q[q_idx] = coeffs[0]
        fitted = A @ coeffs
        ss_res = np.sum((rv - fitted) ** 2)
        ss_tot = np.sum((rv - rv.mean()) ** 2)
        R2[q_idx] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "q_values": q_values,
        "zeta_q": zeta_q,
        "R2": R2,
        "j_levels": j_vals,
        "log2_Mq": log2_Mq,
    }


def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path("data/processed")
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    wl_cfg = cfg["wavelet"]
    q_values = np.linspace(wl_cfg["q_min"], wl_cfg["q_max"], wl_cfg["q_steps"])

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
        if len(rv) < 64:
            continue

        log.info("Wavelet leaders: %s", label)

        result = wavelet_scaling_exponents(
            rv,
            q_values,
            wavelet=wl_cfg["wavelet"],
            min_j=wl_cfg["min_scale_j"],
            max_j=wl_cfg["max_scale_j"],
        )

        zeta = result["zeta_q"]
        finite = zeta[np.isfinite(zeta)]
        sw = float(finite.max() - finite.min()) if len(finite) >= 2 else np.nan

        rows.append(
            {
                "exchange": exchange,
                "asset": asset,
                "frequency": freq,
                "year": year,
                "spectral_width_zeta": sw,
                "zeta_q_min": float(np.nanmin(zeta)),
                "zeta_q_max": float(np.nanmax(zeta)),
                "mean_R2": float(np.nanmean(result["R2"])),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = tables_dir / "wavelet_leaders_summary.csv"
    out_df.to_csv(out_path, index=False)
    log.info("Wavelet leaders summary → %s", out_path)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Wavelet leaders multifractal analysis")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
