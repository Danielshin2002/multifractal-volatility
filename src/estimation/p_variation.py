"""
Cont-Das normalised p-variation estimator.

Mathematical reference: Pontiggia (2025) arXiv:2507.00575v3, §3.1 and §3.4-3.5.

Exact formula (eq. 2 / eq. 8 in paper):

    Given X = {X_t}^L_{t=1}, partition into K consecutive non-overlapping
    blocks of size n = L/K.  Block sums:

        S_j = sum_{i=(j-1)*n + 1}^{j*n}  X_i,   j = 1, ..., K

    Normalised p-variation statistic:

        W(L, K, p) = (1/K) * sum_{j=1}^{K} |S_j|^p          (eq. 2)

    Roughness index:

        H_hat = 1 / p*   where   log W(L, K, p*) = 0         (eq. 3)

K selection (§3.4):

    K_opt = n_opt = floor(sqrt(N))
    L_used = K_opt^2   (trim to exact multiple; discards at most ~2*sqrt(N) obs)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_W(series: NDArray[np.float64], K: int, p: float) -> float:
    """
    Compute the normalised p-variation statistic W(L, K, p).

    Parameters
    ----------
    series:
        Realized-volatility (or any) time series, shape (N,).
    K:
        Number of blocks.
    p:
        Moment order (p > 0).

    Returns
    -------
    float
        W(L, K, p) = (1/K) * sum_j |S_j|^p, where S_j are non-overlapping
        block sums of size n = L // K.

    Raises
    ------
    ValueError
        If K >= len(series) or p <= 0.
    """
    if p <= 0:
        raise ValueError(f"p must be > 0, got {p}")
    N = len(series)
    if K < 1 or K >= N:
        raise ValueError(f"K must satisfy 1 <= K < N={N}, got K={K}")

    n = N // K          # block size
    L = K * n           # trim to exact divisible length
    trimmed = series[:L]

    # Block sums S_j, j = 1, ..., K  (shape: (K,))
    blocks = trimmed.reshape(K, n)
    block_sums = blocks.sum(axis=1)

    return float(np.mean(np.abs(block_sums) ** p))


def k_opt(N: int) -> int:
    """
    Optimal number of blocks following Cont-Das (2024) recommendation.

    K_opt = n_opt = floor(sqrt(N))

    The block size equals K, giving L = K^2 observations used in estimation.
    This minimises the bias-variance trade-off in finite samples.
    """
    return int(np.floor(np.sqrt(N)))


def log_W_curve(
    series: NDArray[np.float64],
    K: int,
    p_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate log W(L, K, p) for each p in p_values.

    Returns
    -------
    NDArray of shape (len(p_values), 2)
        Columns: [1/p,  log W(L, K, p)].
        Rows where W == 0 are excluded.
    """
    rows = []
    for p in p_values:
        W = compute_W(series, K, p)
        if W > 0:
            rows.append((1.0 / p, np.log(W)))
    return np.array(rows, dtype=np.float64)


def find_zero_crossing(curve: NDArray[np.float64]) -> float | None:
    """
    Locate p* such that log W(L, K, p*) = 0 by linear interpolation.

    Parameters
    ----------
    curve:
        Array of shape (M, 2) with columns [1/p, log W].
        Must be sorted by increasing 1/p (i.e. decreasing p).

    Returns
    -------
    float or None
        Roughness index H = 1/p* if a zero-crossing exists, else None.
    """
    if curve.shape[0] < 2:
        return None

    inv_p = curve[:, 0]
    log_w = curve[:, 1]

    sign_changes = np.where(np.diff(np.sign(log_w)))[0]
    if len(sign_changes) == 0:
        return None

    # First zero-crossing (smallest 1/p where sign flips, i.e. largest p*)
    i = sign_changes[0]
    # Linear interpolation: find inv_p where log_w = 0
    inv_p_star = inv_p[i] + (0.0 - log_w[i]) * (inv_p[i + 1] - inv_p[i]) / (
        log_w[i + 1] - log_w[i]
    )
    p_star = 1.0 / inv_p_star
    return float(1.0 / p_star)


def estimate_roughness(
    series: NDArray[np.float64],
    K: int | None = None,
    p_min: float = 0.1,
    p_max: float = 4.0,
    n_steps: int = 100,
) -> dict:
    """
    Estimate roughness index from a realized-volatility series.

    Parameters
    ----------
    series:
        1-D array of realized volatility values (e.g. absolute returns).
    K:
        Number of blocks.  Defaults to k_opt(len(series)).
    p_min, p_max, n_steps:
        p-grid specification.  Paper uses [0.1, 4.0] (standard) or
        [0.01, 4.0] (wide).

    Returns
    -------
    dict with keys:
        K, n, L, p_grid, curve (shape M×2), H (float or None),
        log_W_min, log_W_max
    """
    N = len(series)
    if K is None:
        K = k_opt(N)

    p_grid = np.linspace(p_min, p_max, n_steps)
    curve = log_W_curve(series, K, p_grid)

    H = find_zero_crossing(curve) if curve.shape[0] >= 2 else None

    n_block = N // K
    L = K * n_block

    return {
        "K": K,
        "n": n_block,
        "L": L,
        "p_grid": p_grid,
        "curve": curve,
        "H": H,
        "log_W_min": float(curve[:, 1].min()) if curve.shape[0] > 0 else np.nan,
        "log_W_max": float(curve[:, 1].max()) if curve.shape[0] > 0 else np.nan,
    }


def roughness_vs_K(
    series: NDArray[np.float64],
    K_values: NDArray[np.int_],
    p_min: float = 0.1,
    p_max: float = 4.0,
    n_steps: int = 100,
) -> NDArray[np.float64]:
    """
    Compute H_hat(K) over a range of K values for diagnostic plotting.

    Returns
    -------
    NDArray of shape (len(K_values), 2)
        Columns: [K, H_hat].  H_hat is NaN where no zero-crossing exists.
    """
    p_grid = np.linspace(p_min, p_max, n_steps)
    results = []
    for K in K_values:
        if K < 1 or K >= len(series):
            continue
        curve = log_W_curve(series, int(K), p_grid)
        H = find_zero_crossing(curve) if curve.shape[0] >= 2 else None
        results.append((float(K), float(H) if H is not None else np.nan))
    return np.array(results, dtype=np.float64)
