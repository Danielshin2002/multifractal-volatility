"""
Fetch raw OHLCV data via CCXT for all (exchange, asset, frequency) combos.

Saves compressed parquet files to data/raw/{exchange}/{asset}/{freq}.parquet.
No API keys required for public historical data on all supported exchanges.

Usage:
    python src/data/fetch.py --config config.yaml [--exchange binanceus] [--asset BTC/USDT]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import ccxt
import pandas as pd
import yaml

log = logging.getLogger(__name__)

# CCXT uses millisecond timestamps
MS_PER_SECOND = 1_000
TIMEFRAME_TO_MS = {
    "1m": 60 * MS_PER_SECOND,
    "5m": 5 * 60 * MS_PER_SECOND,
    "10m": 10 * 60 * MS_PER_SECOND,
    "15m": 15 * 60 * MS_PER_SECOND,
}

# Exchanges that list BTC/USDT under a different symbol
SYMBOL_OVERRIDES: dict[str, dict[str, str]] = {
    "coinbase": {"BTC/USDT": "BTC/USD", "ETH/USDT": "ETH/USD"},
    "kraken": {"BTC/USDT": "BTC/USD", "ETH/USDT": "ETH/USD"},
}

# Some exchanges don't natively support 10m; we fetch 1m and resample later
SYNTHETIC_TIMEFRAMES = {"10m"}


def get_exchange(name: str) -> ccxt.Exchange:
    cls = getattr(ccxt, name)
    exchange = cls({"enableRateLimit": True})
    exchange.load_markets()
    return exchange


def resolve_symbol(exchange_name: str, symbol: str) -> str:
    overrides = SYMBOL_OVERRIDES.get(exchange_name, {})
    return overrides.get(symbol, symbol)


def fetch_ohlcv_paginated(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    batch_size: int = 1000,
) -> list[list]:
    """Paginate through OHLCV data between since_ms and until_ms."""
    all_candles: list[list] = []
    cursor = since_ms
    tf_ms = TIMEFRAME_TO_MS.get(timeframe, 60 * MS_PER_SECOND)

    while cursor < until_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=cursor,
                limit=batch_size,
            )
        except ccxt.NetworkError as e:
            log.warning("Network error: %s. Retrying in 10s…", e)
            time.sleep(10)
            continue
        except ccxt.ExchangeError as e:
            log.error("Exchange error: %s. Stopping pagination.", e)
            break

        if not candles:
            break

        first_ts = candles[0][0]
        # Some exchanges (e.g. Kraken) ignore the `since` parameter and always
        # return the most recent candles. Detect this by checking whether the
        # first returned timestamp is far ahead of where we asked to start.
        if first_ts > cursor + batch_size * tf_ms:
            log.warning(
                "%s appears to ignore `since` (asked %s, got %s); "
                "historical pagination not supported for this exchange.",
                exchange.id,
                exchange.iso8601(cursor),
                exchange.iso8601(first_ts),
            )
            break

        all_candles.extend(candles)
        last_ts = candles[-1][0]

        if last_ts >= until_ms:
            break

        cursor = last_ts + tf_ms
        # Respect exchange rate limit
        time.sleep(exchange.rateLimit / MS_PER_SECOND)

    return all_candles


def candles_to_df(candles: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(
        candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_and_save(
    exchange_name: str,
    asset: str,
    timeframe: str,
    start: str,
    end: str,
    out_dir: Path,
) -> None:
    """Fetch one (exchange, asset, timeframe) combination and save to parquet.

    Resume-aware: if a partial parquet already exists, picks up from the last
    saved timestamp rather than restarting from scratch.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # For 10m, fetch 1m and resample (many exchanges lack native 10m support)
    fetch_tf = "1m" if timeframe in SYNTHETIC_TIMEFRAMES else timeframe
    fetch_tf_ms = TIMEFRAME_TO_MS.get(fetch_tf, 60 * MS_PER_SECOND)
    out_path = out_dir / f"{timeframe}.parquet"

    try:
        exchange = get_exchange(exchange_name)
    except Exception as e:
        log.error("Could not load exchange %s: %s", exchange_name, e)
        return

    symbol = resolve_symbol(exchange_name, asset)
    if symbol not in exchange.markets:
        log.warning(
            "Symbol %s not available on %s; skipping", symbol, exchange_name
        )
        return

    since_ms = exchange.parse8601(f"{start}T00:00:00Z")
    until_ms = exchange.parse8601(f"{end}T23:59:59Z")

    # Resume: if a file exists, advance since_ms past the last saved candle
    existing: pd.DataFrame | None = None
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        last_saved_ts = existing.index.max()
        last_saved_ms = int(last_saved_ts.timestamp() * MS_PER_SECOND)
        resume_since_ms = last_saved_ms + fetch_tf_ms

        if resume_since_ms >= until_ms:
            log.info(
                "%s/%s/%s complete (%d rows); skipping",
                exchange_name, asset, timeframe, len(existing),
            )
            return

        log.info(
            "%s/%s/%s resuming from %s (%d rows already saved)",
            exchange_name, asset, timeframe, last_saved_ts, len(existing),
        )
        since_ms = resume_since_ms

    log.info(
        "Fetching %s %s %s from %s",
        exchange_name, symbol, fetch_tf,
        pd.Timestamp(since_ms, unit="ms", tz="UTC"),
    )
    candles = fetch_ohlcv_paginated(
        exchange, symbol, fetch_tf, since_ms, until_ms
    )

    if not candles:
        log.warning("No new candles returned for %s %s %s", exchange_name, symbol, fetch_tf)
        return

    new_df = candles_to_df(candles)

    # Resample 1m → 10m if needed
    if timeframe in SYNTHETIC_TIMEFRAMES and fetch_tf == "1m":
        new_df = (
            new_df.resample("10min")
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .dropna(subset=["close"])
        )

    # Merge with existing data when resuming
    if existing is not None:
        df = pd.concat([existing, new_df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = new_df

    df.to_parquet(out_path, compression="gzip")
    log.info("Saved %d rows to %s", len(df), out_path)


def run(config_path: str, exchange_filter: str | None, asset_filter: str | None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exchanges = cfg["exchanges"]
    assets = cfg["assets"]
    frequencies = cfg["frequencies"]
    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]

    if exchange_filter:
        exchanges = [e for e in exchanges if e == exchange_filter]
    if asset_filter:
        assets = [a for a in assets if a == asset_filter]

    raw_dir = Path("data/raw")

    for exchange_name in exchanges:
        for asset in assets:
            asset_slug = asset.replace("/", "_")
            for freq in frequencies:
                out_dir = raw_dir / exchange_name / asset_slug
                fetch_and_save(exchange_name, asset, freq, start, end, out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch OHLCV data via CCXT")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--exchange", default=None, help="Filter to one exchange")
    parser.add_argument("--asset", default=None, help="Filter to one asset")
    args = parser.parse_args()
    run(args.config, args.exchange, args.asset)
