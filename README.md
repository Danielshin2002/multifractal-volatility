# Multifractality & Rough Volatility: Cross-Exchange Replication Study

Replicates and extends Pontiggia (2025) *"Multifractality in Bitcoin Realized Volatility: Implications for Rough Volatility Modelling"* ([arXiv:2507.00575v3](https://arxiv.org/abs/2507.00575)) by testing whether the paper's findings are robust across the top 5 spot exchanges (Binance, OKX, Bybit, Coinbase, Kraken) and whether they extend to Ethereum.

## Quick Start

```bash
conda env create -f environment.yml
conda activate multifractal-vol

python src/data/fetch.py --config config.yaml
python src/data/preprocess.py --config config.yaml
python src/data/realized_vol.py --config config.yaml
python src/estimation/roughness.py --config config.yaml

# Or run everything:
bash run_pipeline.sh
```

## Repository Structure

```
multifractal-volatility/
├── data/
│   ├── raw/                    # Raw OHLCV parquet files per exchange/asset
│   └── processed/              # Cleaned realized volatility series
├── src/
│   ├── data/
│   │   ├── fetch.py            # CCXT data fetching with pagination
│   │   ├── preprocess.py       # 90-day window selection, completeness filter
│   │   └── realized_vol.py     # RV_t = |r_t| and noise-robust variant
│   ├── estimation/
│   │   ├── p_variation.py      # Cont-Das W(L,K,p) estimator
│   │   └── roughness.py        # Roughness index and K-stability diagnostics
│   ├── diagnostics/
│   │   ├── stationarity.py     # ADF, rolling stability, structural breaks
│   │   ├── mfdfa.py            # MF-DFA + shuffle control
│   │   ├── moment_scaling.py   # Log-log moment scaling (zeta_q)
│   │   └── wavelet_leaders.py  # Wavelet leaders multifractal analysis
│   └── visualization/
│       ├── p_variation_plots.py
│       ├── multifractal_plots.py
│       └── comparison_plots.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_roughness_estimation.ipynb
│   ├── 03_multifractality.ipynb
│   └── 04_cross_exchange_comparison.ipynb
├── results/
│   ├── tables/
│   └── figures/
├── tests/
│   └── test_estimators.py
├── config.yaml
├── requirements.txt
├── environment.yml
└── run_pipeline.sh
```

## Key Methods

### Cont-Das p-Variation Estimator

For series `X` with `L` observations, partition into `K` blocks of size `n = L/K`:

```
S_j = sum_{i=(j-1)n+1}^{jn} X_i          # block sum
W(L, K, p) = (1/K) * sum_{j=1}^K |S_j|^p  # normalised p-variation
```

`K_opt = floor(sqrt(N))` (Cont-Das recommendation). The roughness index `H = 1/p*` where `log W(L, K, p*) = 0`.

### MF-DFA

Generalised Hurst exponent `H(q)` estimated via `log F_q(s) ~ H(q) log s`. Curvature in `H(q)` vs `q` signals multifractality. Shuffle control removes temporal dependence to isolate dynamic from distributional multifractality.

## Findings (fill in after running)

*TBD*

## Citation

```
Pontiggia, M. (2025). Multifractality in Bitcoin Realized Volatility: Implications
for Rough Volatility Modelling. arXiv:2507.00575v3.
```
