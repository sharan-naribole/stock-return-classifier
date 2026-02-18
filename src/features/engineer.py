"""
Feature engineer — creates technical indicators for stock return classification.
Uses the `ta` library.

Full feature set for EDA (22 features):
  Price:      Close, Volume, VIX_Close
  Bollinger:  BB_High, BB_Low, BB_Width, BB_Position
  Trend:      EMA_8, EMA_21, ADX
  Momentum:   RSI, MACD_line, MACD_signal, MACD_hist,
              Stoch_K, Stoch_D, ROC_3, ROC_5,
              Price_Return_1, Price_Return_5
  Volatility: IBS, ATR_pct

No transforms are applied here. EDA will identify which features are skewed
(candidates for log1p), redundant (high inter-feature correlation), or
uninformative (low MI / correlation with target). Those findings will drive
the final feature set used for ML training.

Note: High and Low are fetched from yfinance and used internally to compute
IBS, ATR_pct, Stoch_K/D, and ADX. They are NOT passed as model features
(r≈0.98 with Close; their signal is captured by the derived indicators).
"""

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator


class FeatureEngineer:
    """
    Creates technical indicator features from raw OHLCV + VIX data.

    Full feature set (22 features):
      Close, Volume, VIX_Close,
      BB_High, BB_Low, BB_Width, BB_Position,
      EMA_8, EMA_21, ADX,
      RSI, MACD_line, MACD_signal, MACD_hist,
      Stoch_K, Stoch_D, ROC_3, ROC_5,
      Price_Return_1, Price_Return_5,
      IBS, ATR_pct
    """

    def __init__(self, config):
        self.config = config
        features = config.get("data.features") or {}
        self.bollinger_period = features.get("bollinger_period", 20)
        self.bollinger_std   = features.get("bollinger_std", 2)
        self.ema_short       = features.get("ema_short", 8)
        self.ema_medium      = features.get("ema_medium", 21)
        self.adx_period      = features.get("adx_period", 14)
        self.rsi_period      = features.get("rsi_period", 14)
        self.macd_fast       = features.get("macd_fast", 12)
        self.macd_slow       = features.get("macd_slow", 26)
        self.macd_signal     = features.get("macd_signal", 9)
        self.stoch_period    = features.get("stoch_period", 14)
        self.stoch_smooth    = features.get("stoch_smooth", 3)
        self.roc_short       = features.get("roc_short", 3)
        self.roc_long        = features.get("roc_long", 5)
        self.atr_period      = features.get("atr_period", 14)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe.
        Expects columns: Adj_Close (or Close), High, Low, Volume, VIX_Close.
        High and Low are used internally for IBS, ATR, ADX, Stochastic — not model features.
        Returns df with all features added and NaN rows dropped.
        """
        df = df.copy()

        # Standardize close column name
        if "Adj_Close" in df.columns:
            df["Close"] = df["Adj_Close"]
        elif "Close" not in df.columns:
            raise ValueError("DataFrame must have 'Adj_Close' or 'Close' column")

        close = df["Close"]
        high  = df["High"] if "High" in df.columns else close
        low   = df["Low"]  if "Low"  in df.columns else close

        # ── Bollinger Bands ────────────────────────────────────────────────────
        bb = BollingerBands(close=close, window=self.bollinger_period, window_dev=self.bollinger_std)
        df["BB_High"]     = bb.bollinger_hband()
        df["BB_Low"]      = bb.bollinger_lband()
        bb_mid            = bb.bollinger_mavg()
        raw_bb_width      = (df["BB_High"] - df["BB_Low"]) / bb_mid.replace(0, float("nan"))
        df["BB_Width"]    = raw_bb_width
        df["BB_Position"] = (close - df["BB_Low"]) / (df["BB_High"] - df["BB_Low"]).replace(0, float("nan"))

        # ── Trend ──────────────────────────────────────────────────────────────
        df[f"EMA_{self.ema_short}"]  = EMAIndicator(close=close, window=self.ema_short).ema_indicator()
        df[f"EMA_{self.ema_medium}"] = EMAIndicator(close=close, window=self.ema_medium).ema_indicator()
        df["ADX"] = ADXIndicator(high=high, low=low, close=close, window=self.adx_period).adx()

        # ── Momentum ───────────────────────────────────────────────────────────
        df["RSI"] = RSIIndicator(close=close, window=self.rsi_period).rsi()

        macd = MACD(close=close, window_fast=self.macd_fast, window_slow=self.macd_slow, window_sign=self.macd_signal)
        df["MACD_line"]   = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"]   = macd.macd_diff()

        stoch = StochasticOscillator(high=high, low=low, close=close, window=self.stoch_period, smooth_window=self.stoch_smooth)
        # Clip to [0, 100]: Adj_Close outside raw High/Low range causes out-of-range values
        df["Stoch_K"] = stoch.stoch().clip(0, 100)
        df["Stoch_D"] = stoch.stoch_signal().clip(0, 100)

        df[f"ROC_{self.roc_short}"] = ROCIndicator(close=close, window=self.roc_short).roc()
        df[f"ROC_{self.roc_long}"]  = ROCIndicator(close=close, window=self.roc_long).roc()

        df["Price_Return_1"] = close.pct_change(1) * 100
        df["Price_Return_5"] = close.pct_change(5) * 100

        # ── IBS — Intra-Day Bar Strength ───────────────────────────────────────
        # IBS = (Close − Low) / (High − Low)
        # Near 1 = closed at top of day's range (bullish), near 0 = bottom (bearish)
        # Clip to [0, 1]: Adj_Close can fall outside raw High/Low after split/dividend
        # adjustments, producing out-of-range values that are data artifacts.
        bar_range = (high - low).replace(0, float("nan"))
        df["IBS"] = ((close - low) / bar_range).clip(0, 1)

        # ── Volatility ─────────────────────────────────────────────────────────
        # ATR_pct = ATR / Close * 100 (scale-invariant)
        atr = AverageTrueRange(high=high, low=low, close=close, window=self.atr_period).average_true_range()
        df["ATR_pct"] = atr / close * 100

        # Drop raw H/L — used only for derived indicators, not model features
        df.drop(columns=[c for c in ["High", "Low", "Adj_Close"] if c in df.columns], inplace=True)

        # Drop rows with NaN (indicator warm-up period)
        df.dropna(inplace=True)

        return df

    def get_feature_names(self) -> list:
        """Return list of all model feature column names (22 features)."""
        return [
            "Close", "Volume", "VIX_Close",
            "BB_High", "BB_Low", "BB_Width", "BB_Position",
            f"EMA_{self.ema_short}", f"EMA_{self.ema_medium}", "ADX",
            "RSI", "MACD_line", "MACD_signal", "MACD_hist",
            "Stoch_K", "Stoch_D",
            f"ROC_{self.roc_short}", f"ROC_{self.roc_long}",
            "Price_Return_1", "Price_Return_5",
            "IBS", "ATR_pct",
        ]
