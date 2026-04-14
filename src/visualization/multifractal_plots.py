"""
Multifractality diagnostic plots.

Generates:
  1. H(q) vs q spectra (MF-DFA) with shuffle control overlay
  2. ζ_q vs q curves (moment scaling and wavelet leaders)

Usage:
    python src/visualization/multifractal_plots.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.diagnostics.mfdfa import hurst_exponents, shuffle_control
from src.diagnostics.moment_scaling import scaling_exponents
from src.diagnostics.wavelet_leaders import wavelet_scaling_exponents

log = logging.getLogger(__name__)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def plot_hq_spectrum(
    rv: np.ndarray,
    label: str,
    cfg: dict,
    out_path: Path,
) -> None:
    mf_cfg = cfg["mfdfa"]
    q_values = np.linspace(mf_cfg["q_min"], mf_cfg["q_max"], mf_cfg["q_steps"])
    hurst_kwargs = dict(
        min_scale=mf_cfg["min_scale"],
        max_scale_pct=mf_cfg["max_scale_pct"],
        poly_order=mf_cfg["poly_order"],
    )

    orig = hurst_exponents(rv, q_values, **hurst_kwargs)
    shuf = shuffle_control(rv, q_values, n_shuffles=mf_cfg["n_shuffles"], **hurst_kwargs)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(q_values, orig["H"], "b-o", ms=4, lw=1.5, label="Original")
    ax.plot(q_values, shuf["H_mean"], "r--s", ms=4, lw=1.5, label="Shuffled (mean)")
    ax.fill_between(
        q_values,
        shuf["H_mean"] - shuf["H_std"],
        shuf["H_mean"] + shuf["H_std"],
        alpha=0.2, color="r",
    )
    ax.set_xlabel("q")
    ax.set_ylabel("H(q)")
    ax.set_title(f"MF-DFA H(q) spectrum  |  {label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_zeta_q(
    rv: np.ndarray,
    label: str,
    cfg: dict,
    out_path: Path,
) -> None:
    ms_cfg = cfg["moment_scaling"]
    wl_cfg = cfg["wavelet"]
    q_values = np.linspace(ms_cfg["q_min"], ms_cfg["q_max"], ms_cfg["q_steps"])

    ms_result = scaling_exponents(
        rv, q_values,
        tau_min=ms_cfg["tau_min"],
        tau_max_pct=ms_cfg["tau_max_pct"],
    )
    wl_result = wavelet_scaling_exponents(
        rv, q_values,
        wavelet=wl_cfg["wavelet"],
        min_j=wl_cfg["min_scale_j"],
        max_j=wl_cfg["max_scale_j"],
    )

    # Linear reference (monofractal: ζ_q = H*q)
    H_mono = 0.5  # Brownian motion reference
    zeta_linear = H_mono * q_values

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, result, title in zip(
        axes,
        [ms_result, wl_result],
        ["Moment scaling", "Wavelet leaders"],
    ):
        ax.plot(q_values, result["zeta_q"], "b-o", ms=4, lw=1.5, label=r"$\zeta_q$ (empirical)")
        ax.plot(q_values, zeta_linear, "k--", lw=1, alpha=0.6, label="Linear (H=0.5)")
        ax.set_xlabel("q")
        ax.set_ylabel(r"$\zeta_q$")
        ax.set_title(f"{title}  |  {label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path("data/processed")
    figures_dir = Path("results/figures/multifractal")

    for fpath in sorted(processed_dir.glob("rv_*.parquet")):
        parts = fpath.stem.split("_")
        if len(parts) < 5:
            continue
        exchange, asset, freq, year = parts[1], parts[2], parts[3], parts[4]
        label = f"{exchange}/{asset}/{freq}/{year}"
        slug = fpath.stem[3:]

        df = pd.read_parquet(fpath)
        if "rv" not in df.columns:
            continue
        rv = df["rv"].dropna().to_numpy(dtype=np.float64)
        if len(rv) < 100:
            continue

        log.info("Multifractal plots: %s", label)

        plot_hq_spectrum(rv, label, cfg, figures_dir / f"Hq_{slug}.png")
        plot_zeta_q(rv, label, cfg, figures_dir / f"zeta_q_{slug}.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)
