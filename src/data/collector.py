"""
Data collector — downloads SPY + VIX data from yfinance with smart caching.
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


class DataCollector:
    """
    Downloads SPY (Adj Close, High, Low, Volume) and VIX (Close) from yfinance.
    Caches results as CSV in data/raw/ using naming: {TICKER}_{START}_{END}.csv
    Adds a 60-day buffer before start_date for indicator warm-up.
    """

    BUFFER_DAYS = 60  # trading days buffer for indicator warm-up (MACD slow=26 + signal=9)

    def __init__(self, config):
        self.config = config
        self.ticker = config.get("ticker") or "SPY"
        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        raw_dir = config.get("output.data_dir") or "data"
        self.raw_dir = Path(raw_dir) / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _buffered_start(self) -> str:
        """Return start date minus ~buffer calendar days (1.4x to account for weekends/holidays)."""
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        buffer = timedelta(days=int(self.BUFFER_DAYS * 1.4))
        buffered = start - buffer
        return buffered.strftime("%Y-%m-%d")

    def _cache_path(self, ticker: str, start: str, end: str) -> Path:
        start_str = start.replace("-", "")
        end_str = end.replace("-", "")
        return self.raw_dir / f"{ticker}_{start_str}_{end_str}.csv"

    def _find_cached_file(self, ticker: str, start: str, end: str) -> Optional[Path]:
        """Find a cached file that covers the requested date range."""
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        pattern = re.compile(rf"^{re.escape(ticker)}_(\d{{8}})_(\d{{8}})\.csv$")

        for f in self.raw_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                cached_start = datetime.strptime(m.group(1), "%Y%m%d")
                cached_end = datetime.strptime(m.group(2), "%Y%m%d")
                if cached_start <= start_dt and cached_end >= end_dt:
                    return f
        return None

    def _download(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Download data from yfinance."""
        print(f"  Downloading {ticker} from {start} to {end}...")
        # auto_adjust=True: High, Low, Close are all consistently adjusted for splits
        # and dividends — ensures IBS and Stochastic are computed on a consistent scale.
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker} ({start} to {end})")
        return df

    def collect_spy(self) -> pd.DataFrame:
        """Collect SPY OHLCV data (Adj Close, High, Low, Volume) with caching."""
        buffered_start = self._buffered_start()
        cache = self._find_cached_file(self.ticker, buffered_start, self.end_date)

        if cache:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            # Invalidate cache if missing High/Low, or if it still has raw "Adj_Close"
            # alongside High/Low (indicates it was downloaded with auto_adjust=False)
            if "High" not in df.columns or "Low" not in df.columns:
                print(f"  Cache outdated (missing High/Low), re-downloading...")
                cache = None
            elif "Adj_Close" in df.columns and df["Adj_Close"].iloc[-1] != df.get("Close", pd.Series([None])).iloc[-1]:
                print(f"  Cache outdated (downloaded with auto_adjust=False), re-downloading...")
                cache = None

        if cache:
            print(f"  Cache hit: {cache.name}")
        else:
            df = self._download(self.ticker, buffered_start, self.end_date)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            cache_path = self._cache_path(self.ticker, buffered_start, self.end_date)
            df.to_csv(cache_path)
            print(f"  Saved to cache: {cache_path.name}")

        # With auto_adjust=True, "Close" is the adjusted close (no separate "Adj Close")
        col_map = {"Close": "Adj_Close", "High": "High", "Low": "Low", "Volume": "Volume"}
        available = {old: new for old, new in col_map.items() if old in df.columns}
        df = df[list(available.keys())].copy()
        df.rename(columns=available, inplace=True)
        df.index.name = "Date"
        return df

    def collect_vix(self) -> pd.DataFrame:
        """Collect VIX data (Close) with caching."""
        buffered_start = self._buffered_start()
        vix_ticker = "^VIX"
        safe_name = "VIX"
        cache = self._find_cached_file(safe_name, buffered_start, self.end_date)

        if cache:
            print(f"  Cache hit: {cache.name}")
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
        else:
            df = self._download(vix_ticker, buffered_start, self.end_date)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            cache_path = self._cache_path(safe_name, buffered_start, self.end_date)
            df.to_csv(cache_path)
            print(f"  Saved to cache: {cache_path.name}")

        df = df[["Close"]].copy()
        df.columns = ["VIX_Close"]
        df.index.name = "Date"
        return df

    def combine_data(self, spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
        """Inner join SPY and VIX on date index."""
        combined = spy_df.join(vix_df, how="inner")
        combined.sort_index(inplace=True)
        return combined

    def collect_data(self) -> pd.DataFrame:
        """Main entry point: collect and combine SPY + VIX data."""
        print(f"Collecting SPY data ({self.start_date} to {self.end_date})...")
        spy_df = self.collect_spy()
        print(f"Collecting VIX data ({self.start_date} to {self.end_date})...")
        vix_df = self.collect_vix()
        combined = self.combine_data(spy_df, vix_df)
        print(f"Combined data: {len(combined)} rows, {len(combined.columns)} columns")
        return combined
