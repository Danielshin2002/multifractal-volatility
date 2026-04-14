"""
Plots for the Cont-Das p-variation estimator (paper Figures 1, 2).

Generates:
  1. log W(L,K,p) vs 1/p curves per (exchange, asset, year, frequency)
  2. H_hat(K) vs K diagnostic plots (paper Figure 1)

Usage:
    python src/visualization/p_variation_plots.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.estimation.p_variation import estimate_roughness, k_opt, roughness_vs_K

log = logging.getLogger(__name__)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def plot_log_W_curve(
    rv: np.ndarray,
    K: int,
    label: str,
    p_grids: dict[str, tuple[float, float]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(p_grids), figsize=(5 * len(p_grids), 4), sharey=False)
    if len(p_grids) == 1:
        axes = [axes]

    for ax, (grid_name, (p_min, p_max)) in zip(axes, p_grids.items()):
        result = estimate_roughness(rv, K=K, p_min=p_min, p_max=p_max, n_steps=100)
        curve = result["curve"]
        if curve.shape[0] == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        ax.plot(curve[:, 0], curve[:, 1], "b-", lw=1.5, label=f"K={K}")
        ax.axhline(0, color="k", lw=0.8, ls="--", label="log W = 0")

        if result["H"] is not None:
            ax.axvline(result["H"], color="r", lw=0.8, ls=":", label=f"H={result['H']:.3f}")

        ax.set_xlabel("1/p")
        ax.set_ylabel("log W(L,K,p)")
        ax.set_title(f"{label}  [{grid_name} grid]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_H_vs_K(
    rv: np.ndarray,
    label: str,
    out_path: Path,
    k_min: int = 50,
    k_max_factor: float = 0.5,
    n_steps: int = 50,
) -> None:
    N = len(rv)
    k_max = max(k_min + 1, int(k_max_factor * N))
    K_values = np.unique(
        np.round(np.logspace(np.log10(k_min), np.log10(k_max), n_steps)).astype(int)
    )
    hk_data = roughness_vs_K(rv, K_values)

    fig, ax = plt.subplots(figsize=(6, 4))
    valid = np.isfinite(hk_data[:, 1])
    ax.plot(hk_data[valid, 0], hk_data[valid, 1], "b-o", ms=3, lw=1)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(k_opt(N), color="r", lw=1, ls=":", label=f"K_opt={k_opt(N)}")
    ax.set_xlabel("Number of blocks K")
    ax.set_ylabel(r"$\hat{H}_{L,K}$")
    ax.set_title(f"Roughness vs K  |  {label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_log_W_btc_vs_eth_2020(
    processed_dir: Path,
    p_min: float,
    p_max: float,
    out_path: Path,
    freq: str = "1m",
) -> None:
    """
    Side-by-side log W(L,K,p) curves for BTC vs ETH in 2020.

    Panels: one per exchange that has both assets in 2020.
    BTC = orange (#f7931a), ETH = blue (#627eea).
    """
    exchanges = ["okx", "coinbase"]
    colors = {"BTC_USDT": "#f7931a", "ETH_USDT": "#627eea",
               "BTC_USD":  "#f7931a", "ETH_USD":  "#627eea"}
    labels  = {"BTC_USDT": "BTC", "ETH_USDT": "ETH",
               "BTC_USD":  "BTC", "ETH_USD":  "ETH"}

    fig, axes = plt.subplots(1, len(exchanges), figsize=(5 * len(exchanges), 4), sharey=False)

    for ax, exchange in zip(axes, exchanges):
        plotted_any = False
        for asset_slug in ("BTC_USDT", "ETH_USDT", "BTC_USD", "ETH_USD"):
            fpath = processed_dir / f"rv_{exchange}_{asset_slug}_{freq}_2020.parquet"
            if not fpath.exists():
                continue

            df = pd.read_parquet(fpath)
            if "rv" not in df.columns:
                continue
            rv = df["rv"].dropna().to_numpy(dtype=np.float64)
            if len(rv) < 100:
                continue

            K = k_opt(len(rv))
            result = estimate_roughness(rv, K=K, p_min=p_min, p_max=p_max, n_steps=100)
            curve = result["curve"]
            if curve.shape[0] == 0:
                continue

            color = colors[asset_slug]
            asset_label = labels[asset_slug]
            ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1.8, label=asset_label)

            if result["H"] is not None:
                ax.axvline(
                    result["H"], color=color, lw=1, ls=":",
                    label=f"{asset_label} H={result['H']:.3f}",
                )
            plotted_any = True

        ax.axhline(0, color="k", lw=0.8, ls="--", label="log W = 0")
        ax.set_xlabel("1/p")
        ax.set_ylabel("log W(L,K,p)")
        ax.set_title(f"{exchange.upper()}  |  {freq}  |  2020")
        if plotted_any:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved %s", out_path)


def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path("data/processed")
    figures_dir = Path("results/figures/p_variation")

    p_std = cfg["estimation"]["p_grid_standard"]
    p_wide = cfg["estimation"]["p_grid_wide"]
    p_grids = {
        "standard": (p_std[0], p_std[1]),
        "wide": (p_wide[0], p_wide[1]),
    }

    for fpath in sorted(processed_dir.glob("rv_*.parquet")):
        parts = fpath.stem.split("_")
        if len(parts) < 6:
            continue
        exchange, asset, freq, year = parts[1], f"{parts[2]}_{parts[3]}", parts[4], parts[5]
        label = f"{exchange}/{asset}/{freq}/{year}"
        slug = fpath.stem[3:]  # strip leading 'rv_'

        df = pd.read_parquet(fpath)
        if "rv" not in df.columns:
            continue
        rv = df["rv"].dropna().to_numpy(dtype=np.float64)
        if len(rv) < 100:
            continue

        K = k_opt(len(rv))
        log.info("Plotting p-variation: %s", label)

        plot_log_W_curve(
            rv, K, label, p_grids,
            figures_dir / f"log_W_{slug}.png"
        )
        plot_H_vs_K(
            rv, label,
            figures_dir / f"H_vs_K_{slug}.png"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--btc-vs-eth-2020",
        action="store_true",
        help="Generate BTC vs ETH 2020 log W comparison plot only",
    )
    parser.add_argument("--freq", default="1m", help="Frequency for --btc-vs-eth-2020")
    args = parser.parse_args()

    if args.btc_vs_eth_2020:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        p_std = cfg["estimation"]["p_grid_standard"]
        plot_log_W_btc_vs_eth_2020(
            processed_dir=Path("data/processed"),
            p_min=p_std[0],
            p_max=p_std[1],
            out_path=Path(f"results/figures/p_variation/log_W_btc_vs_eth_2020_{args.freq}.png"),
            freq=args.freq,
        )
    else:
        run(args.config)
