"""
Cross-exchange and cross-asset comparison plots.

Reads pre-computed summary CSVs from results/tables/ and generates:
  1. MF-DFA spectral width vs exchange (per asset, frequency, year)
  2. Roughness index H (where estimated) across exchanges
  3. BTC vs ETH comparison: spectral width side-by-side
  4. Spectral width vs daily volume (if volume data available)

Usage:
    python src/visualization/comparison_plots.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

EXCHANGE_ORDER = ["binance", "okx", "bybit", "coinbase", "kraken"]
ASSET_COLORS = {"BTC_USDT": "#f7931a", "ETH_USDT": "#627eea",
                "BTC_USD": "#f7931a", "ETH_USD": "#627eea"}


def load_summaries(tables_dir: Path) -> pd.DataFrame:
    """Load and merge p_variation, mfdfa, and moment_scaling summaries."""
    frames = {}
    for name in ["p_variation_summary", "mfdfa_summary", "moment_scaling_summary",
                 "wavelet_leaders_summary"]:
        p = tables_dir / f"{name}.csv"
        if p.exists():
            frames[name] = pd.read_csv(p)

    if not frames:
        return pd.DataFrame()

    base = frames.get("p_variation_summary", list(frames.values())[0])
    keys = ["exchange", "asset", "frequency", "year"]

    for name, df in frames.items():
        if name == "p_variation_summary":
            continue
        suffix = f"_{name.split('_')[0]}"
        drop_cols = [c for c in df.columns if c not in keys]
        df = df.rename(columns={c: c + suffix for c in drop_cols})
        base = base.merge(df, on=keys, how="outer")

    return base


def plot_spectral_width_by_exchange(
    df: pd.DataFrame,
    figures_dir: Path,
    freq_filter: str = "1m",
) -> None:
    sub = df[df["frequency"] == freq_filter].copy()
    if sub.empty:
        log.warning("No data for frequency %s", freq_filter)
        return

    col = "spectral_width_mfdfa" if "spectral_width_mfdfa" in sub else "spectral_width"
    if col not in sub.columns:
        log.warning("Spectral width column not found")
        return

    for asset in sub["asset"].unique():
        sub_asset = sub[sub["asset"] == asset]
        fig, ax = plt.subplots(figsize=(8, 4))
        for year, grp in sub_asset.groupby("year"):
            exchanges = [e for e in EXCHANGE_ORDER if e in grp["exchange"].values]
            values = [
                grp[grp["exchange"] == e][col].mean() for e in exchanges
            ]
            ax.plot(exchanges, values, "o-", ms=6, label=str(year))

        ax.set_xlabel("Exchange")
        ax.set_ylabel("MF-DFA spectral width (max H(q) − min H(q))")
        ax.set_title(f"Spectral width by exchange  |  {asset}  {freq_filter}")
        ax.legend(fontsize=8, title="Year")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = figures_dir / f"spectral_width_{asset}_{freq_filter}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        log.info("Saved %s", out)


def plot_btc_vs_eth(df: pd.DataFrame, figures_dir: Path) -> None:
    col = "spectral_width_mfdfa" if "spectral_width_mfdfa" in df else "spectral_width"
    if col not in df.columns:
        return

    btc = df[df["asset"].str.startswith("BTC")]
    eth = df[df["asset"].str.startswith("ETH")]
    if btc.empty or eth.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for asset_df, color, label in [
        (btc, "#f7931a", "BTC"),
        (eth, "#627eea", "ETH"),
    ]:
        means = asset_df.groupby(["exchange", "frequency"])[col].mean().reset_index()
        ax.scatter(means["frequency"], means[col], c=color, label=label, alpha=0.7, s=40)

    ax.set_xlabel("Frequency")
    ax.set_ylabel("MF-DFA spectral width")
    ax.set_title("BTC vs ETH multifractal spectral width")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = figures_dir / "btc_vs_eth_spectral_width.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_log_W_heatmap(df: pd.DataFrame, figures_dir: Path) -> None:
    col = "log_W_min_std"
    if col not in df.columns:
        return

    for freq in df["frequency"].unique():
        sub = df[df["frequency"] == freq]
        pivot = sub.pivot_table(index="exchange", columns="year", values=col, aggfunc="mean")
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns)), 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-6, vmax=0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label="min log W(L,K,p)")
        ax.set_title(f"min log W heatmap  |  {freq}")
        plt.tight_layout()
        out = figures_dir / f"log_W_heatmap_{freq}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        log.info("Saved %s", out)


def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures/comparison")
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_summaries(tables_dir)
    if df.empty:
        log.warning("No summary tables found — run the analysis pipeline first")
        return

    for freq in cfg["frequencies"]:
        plot_spectral_width_by_exchange(df, figures_dir, freq_filter=freq)

    plot_btc_vs_eth(df, figures_dir)
    plot_log_W_heatmap(df, figures_dir)
    log.info("Comparison plots complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
