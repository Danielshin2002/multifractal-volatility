# Multifractality & Rough Volatility: Cross-Exchange Replication Study
## BTC and ETH across Top 5 Exchanges

This repository replicates and extends Pontiggia (2025) *"Multifractality in Bitcoin 
Realized Volatility: Implications for Rough Volatility Modelling"* by testing whether 
the paper's findings are robust across major high-volume exchanges and whether they 
extend to Ethereum.

---

## Repository Structure

```
multifractal-volatility/
│
├── data/
│   ├── raw/                        # Raw OHLCV data per exchange/asset
│   └── processed/                  # Cleaned realized volatility series
│
├── src/
│   ├── data/
│   │   ├── fetch.py                # Data fetching via CCXT
│   │   ├── preprocess.py           # Cleaning, resampling, completeness checks
│   │   └── realized_vol.py         # Realized volatility construction
│   │
│   ├── estimation/
│   │   ├── p_variation.py          # Cont-Das normalised p-variation estimator
│   │   └── roughness.py            # Roughness index computation and diagnostics
│   │
│   ├── diagnostics/
│   │   ├── stationarity.py         # ADF tests, rolling stability, structural breaks
│   │   ├── mfdfa.py                # Multifractal Detrended Fluctuation Analysis
│   │   ├── moment_scaling.py       # Log-log moment scaling (zeta_q)
│   │   └── wavelet_leaders.py      # Wavelet leaders multifractal analysis
│   │
│   └── visualization/
│       ├── p_variation_plots.py    # log W(L,K,p) curves
│       ├── multifractal_plots.py   # H(q) spectra, zeta_q curves
│       └── comparison_plots.py     # Cross-exchange / cross-asset comparisons
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_roughness_estimation.ipynb
│   ├── 03_multifractality.ipynb
│   └── 04_cross_exchange_comparison.ipynb
│
├── results/
│   ├── tables/                     # CSV outputs of all summary statistics
│   └── figures/                    # All generated plots
│
├── tests/
│   └── test_estimators.py          # Unit tests for p-variation and MF-DFA
│
├── requirements.txt
├── environment.yml
├── config.yaml                     # Central config for exchanges, assets, params
└── README.md
```

---

## Research Design

### Assets
- **BTC/USDT** (or BTC/USD where USDT unavailable)
- **ETH/USDT** (or ETH/USD where USDT unavailable)

### Exchanges (Top 5 by 24h spot volume)
Verify current rankings at [Messari](https://messari.io) or [CoinGecko](https://coingecko.com) 
before fetching, as rankings shift. As of mid-2025, likely candidates are:

1. Binance
2. OKX
3. Bybit
4. Coinbase
5. Kraken

### Sampling Frequencies
Matching the paper: **1-min, 5-min, 10-min, 15-min**

### Time Period
**2020–2024** (5 years). Pre-2020 data quality on non-Bitstamp exchanges is inconsistent.
Use the same 90-day best-window selection method as the paper.

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- Git
- A GitHub account
- (Optional but recommended) Conda or virtualenv

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/multifractal-volatility.git
cd multifractal-volatility
```

### 3. Create Environment

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate multifractal-vol
```

Or using pip:
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 4. Key Dependencies

```
ccxt>=4.0.0              # Unified crypto exchange API
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
statsmodels>=0.14.0      # ADF tests
PyWavelets>=1.4.0        # Wavelet leaders
ruptures>=1.1.7          # Structural break detection
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
pytest>=7.0.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

## Data Fetching

### Using CCXT

CCXT gives you a unified interface across all major exchanges. No API keys are needed 
for historical public OHLCV data on most exchanges.

```python
import ccxt

# Example: fetch 1-minute BTC/USDT from Binance
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv(
    symbol='BTC/USDT',
    timeframe='1m',
    since=exchange.parse8601('2020-01-01T00:00:00Z'),
    limit=1000
)
```

**Important**: Most exchanges limit you to 500–1000 candles per request. You will need 
to paginate — loop with `since` set to the last timestamp retrieved. The `fetch.py` 
script handles this automatically.

### Rate Limits

Be respectful of rate limits. CCXT handles throttling automatically if you use 
`exchange.rateLimit` and `time.sleep()`. Fetching 5 years of 1-minute data per 
exchange/asset pair will take time — plan for **several hours** of data collection 
per pair. Run overnight.

### Data Volume Estimate

Per (exchange, asset, frequency) combination over 5 years:
- 1-min: ~2.6M rows
- 5-min: ~525K rows
- 10-min: ~262K rows
- 15-min: ~175K rows

Across 5 exchanges × 2 assets × 4 frequencies = 40 combinations. 
Store raw data as compressed parquet files to save space.

### config.yaml

```yaml
exchanges:
  - binance
  - okx
  - bybit
  - coinbase
  - kraken

assets:
  - BTC/USDT
  - ETH/USDT

frequencies:
  - 1m
  - 5m
  - 10m
  - 15m

date_range:
  start: "2020-01-01"
  end: "2024-12-31"

estimation:
  window_days: 90
  completeness_threshold: 0.90
  k_rule: "sqrt_n"           # K = floor(sqrt(N))
  p_grid_standard: [0.1, 4.0]
  p_grid_wide: [0.01, 4.0]
  p_grid_steps: 100
  moment_orders: [-4, -3, -2, -1, 0, 1, 2, 3, 4]  # q values for MF-DFA etc.

stationarity:
  adf_significance: 0.05
  rolling_window_pct: 0.05
  max_breakpoints: 5
```

---

## Running the Analysis

### Step 1: Fetch Data
```bash
python src/data/fetch.py --config config.yaml
```
This fetches all OHLCV data and saves to `data/raw/`.

### Step 2: Preprocess
```bash
python src/data/preprocess.py --config config.yaml
```
This applies the 90-day window selection, completeness filtering, and resampling.
Outputs completeness tables to `results/tables/completeness.csv`.

### Step 3: Construct Realized Volatility
```bash
python src/data/realized_vol.py --config config.yaml
```
Constructs both RV_t = |r_t| and the noise-robust RV^Δ_t.

### Step 4: Roughness Estimation
```bash
python src/estimation/roughness.py --config config.yaml
```
Computes W(L,K,p) across all (exchange, asset, year, frequency) combinations.
Outputs log W tables to `results/tables/p_variation_summary.csv`.

### Step 5: Stationarity Diagnostics
```bash
python src/diagnostics/stationarity.py --config config.yaml
```
Runs ADF tests, rolling stability, and structural break detection.

### Step 6: Multifractality Diagnostics
```bash
python src/diagnostics/mfdfa.py --config config.yaml
python src/diagnostics/moment_scaling.py --config config.yaml
python src/diagnostics/wavelet_leaders.py --config config.yaml
```

### Step 7: Generate All Figures
```bash
python src/visualization/p_variation_plots.py --config config.yaml
python src/visualization/multifractal_plots.py --config config.yaml
python src/visualization/comparison_plots.py --config config.yaml
```

### Run Everything (Pipeline)
```bash
bash run_pipeline.sh
```

---

## Key Implementation Details

### Cont-Das p-Variation Estimator

```python
import numpy as np

def compute_W(series, K, p):
    """
    Compute normalised p-variation statistic W(L, K, p).
    
    Parameters
    ----------
    series : np.array  - realized volatility series (absolute returns)
    K      : int       - number of blocks
    p      : float     - moment order
    
    Returns
    -------
    float : W(L, K, p)
    """
    n = len(series) // K          # block size
    L = K * n                     # trim to exact multiple
    trimmed = series[:L]
    blocks = trimmed.reshape(K, n)
    block_sums = np.sum(blocks, axis=1)   # S_j for each block
    W = np.mean(np.abs(block_sums) ** p)
    return W

def compute_log_W_curve(series, K, p_grid):
    """
    Compute log W(L,K,p) over a grid of p values.
    Returns array of (1/p, log W) pairs for plotting.
    """
    results = []
    for p in p_grid:
        W = compute_W(series, K, p)
        if W > 0:
            results.append((1/p, np.log(W)))
    return np.array(results)

def find_zero_crossing(log_W_curve):
    """
    Find p* where log W crosses zero.
    Returns roughness index H = 1/p*, or None if no crossing found.
    """
    inv_p = log_W_curve[:, 0]
    log_W = log_W_curve[:, 1]
    # find sign changes
    sign_changes = np.where(np.diff(np.sign(log_W)))[0]
    if len(sign_changes) == 0:
        return None
    # interpolate at first crossing
    i = sign_changes[0]
    inv_p_star = np.interp(0, [log_W[i], log_W[i+1]], [inv_p[i], inv_p[i+1]])
    p_star = 1 / inv_p_star
    H = 1 / p_star
    return H
```

### MF-DFA (key concept)

MF-DFA estimates a generalised Hurst exponent H(q) for each moment order q. 
Curvature in H(q) vs q signals multifractality. The shuffle control (randomly 
permuting the time series) removes temporal dependence — if H(q) spectrum 
width collapses after shuffling, multifractality is temporal in origin.

Use the `fathon` library or implement from scratch following 
Kantelhardt et al. (2002).

---

## What to Look For: Interpreting Results

### If the paper's finding replicates across exchanges:
- log W(L,K,p) remains strictly negative on Binance, OKX, etc.
- H(q) spectra remain non-linear
- Confirms multifractality is a **structural property of Bitcoin**, not a 
  Bitstamp liquidity artifact

### If higher-volume exchanges show different behavior:
- log W approaches or crosses zero on Binance
- H(q) spectra flatten toward monofractality
- Suggests exchange liquidity **does** affect roughness estimation — 
  the paper's finding is partly a data artifact

### BTC vs ETH comparison:
- If ETH shows similar multifractality: property is shared across major 
  cryptocurrencies
- If ETH shows weaker multifractality or a valid roughness index: 
  market microstructure differences matter

### Cross-exchange comparison metric:
Compute the **MF-DFA spectral width** (max H(q) − min H(q)) per exchange 
and plot against that exchange's average daily volume. A negative correlation 
would directly support the liquidity hypothesis.

---

## Using Claude Code

Claude Code is the recommended tool for building this project. It is a 
command-line AI coding agent that can write, run, debug, and iterate on 
your code directly.

### Installation
```bash
npm install -g @anthropic-ai/claude-code
```

### Recommended Workflow with Claude Code

**1. Scaffold the repository**
```bash
claude "Create the full repository structure for this project as described 
in INSTRUCTIONS.md, including all Python files with function stubs, 
requirements.txt, environment.yml, config.yaml, and run_pipeline.sh"
```

**2. Implement the p-variation estimator**
```bash
claude "Implement src/estimation/p_variation.py with the Cont-Das normalised 
p-variation estimator. Include the W(L,K,p) computation, log W curve generation, 
zero-crossing detection, and K_opt = floor(sqrt(N)) selection. Add unit tests 
in tests/test_estimators.py that verify the estimator returns H≈0.1 for 
simulated fractional Brownian motion with H=0.1"
```

**3. Implement data fetching**
```bash
claude "Implement src/data/fetch.py using ccxt to fetch 1-minute OHLCV data 
for BTC/USDT and ETH/USDT from binance, okx, bybit, coinbase, and kraken. 
Handle pagination, rate limits, missing data, and save as compressed parquet 
files to data/raw/"
```

**4. Debug iteratively**

Claude Code can read error messages and fix them directly. If a script fails:
```bash
claude "The fetch script is failing with [error]. Fix it."
```

**5. Generate analysis and figures**
```bash
claude "Run the full pipeline for Binance BTC/USDT 2021 data and generate 
the log W(L,K,p) plot. Compare it to the Bitstamp result in the paper."
```

### Tips for Working with Claude Code
- Give it one task at a time — scaffold first, then implement piece by piece
- Ask it to write tests alongside the implementation
- Ask it to explain any code it writes that you don't understand
- Commit working code to git frequently so you can roll back if needed
- Ask it to add docstrings and comments as it goes

---

## Setting Up the GitHub Repository

```bash
# Initialize
git init
git add .
git commit -m "Initial scaffold"

# Create repo on GitHub (via web or GitHub CLI)
gh repo create multifractal-volatility --public
git remote add origin https://github.com/YOUR_USERNAME/multifractal-volatility.git
git push -u origin main
```

Recommended branch structure:
- `main` — stable, tested code only
- `dev` — active development
- `feature/data-fetching`, `feature/p-variation`, etc. — individual features

---

## Suggested README Structure for the Repository

Your repository README.md should include:
1. One-paragraph description of the project and its motivation
2. Link to the original Pontiggia (2025) paper
3. Quick start (setup + run pipeline in ~5 commands)
4. Summary of findings (fill in after running)
5. Repository structure overview
6. Citation

---

## References

- Pontiggia, M. (2025). *Multifractality in Bitcoin Realized Volatility: 
  Implications for Rough Volatility Modelling*. arXiv:2507.00575v3.
- Cont, R. and Das, P. (2024). *Rough volatility: Fact or artefact?* 
  Sankhyā B: The Indian Journal of Statistics.
- Kantelhardt, J.W. et al. (2002). *Multifractal detrended fluctuation 
  analysis of nonstationary time series*. Physica A, 316(1–4):87–114.
- Gatheral, J., Jaisson, T., and Rosenbaum, M. (2018). *Volatility is rough*. 
  Quantitative Finance, 18(6):933–949.
