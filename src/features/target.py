"""
Target creator — N-day forward return binary classification target with configurable threshold.
"""

import pandas as pd


class TargetCreator:
    """
    Creates binary classification target based on forward return over horizon_days.

    target = 1  if  (Close[t+horizon] - Close[t]) / Close[t] >= return_threshold_pct / 100
           = 0  otherwise

    Default threshold is 0.0 (any positive return). Set return_threshold_pct in config
    to require a minimum gain, e.g. 1.0 for +1% in 3 days.
    """

    def __init__(self, horizon_days: int = 3, return_threshold_pct: float = 0.0):
        self.horizon_days = horizon_days
        self.return_threshold_pct = return_threshold_pct

    @classmethod
    def from_config(cls, config) -> "TargetCreator":
        horizon = config.get("target.horizon_days") or 3
        threshold = config.get("target.return_threshold_pct")
        if threshold is None:
            threshold = 0.0
        return cls(horizon_days=horizon, return_threshold_pct=float(threshold))

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'target' column to df.
        Rows without a valid future close (last horizon_days rows) are dropped.
        """
        df = df.copy()
        close = df["Close"]
        future_close = close.shift(-self.horizon_days)
        forward_return = (future_close - close) / close * 100
        df["forward_return"] = forward_return
        df["target"] = (forward_return >= self.return_threshold_pct).astype(int)
        df = df.iloc[:-self.horizon_days].copy()
        return df

    def get_target_info(self, df: pd.DataFrame) -> dict:
        """Return class balance info for the target column."""
        counts = df["target"].value_counts()
        total = len(df)
        return {
            "total": total,
            "threshold_pct": self.return_threshold_pct,
            "class_0": int(counts.get(0, 0)),
            "class_1": int(counts.get(1, 0)),
            "class_0_pct": round(counts.get(0, 0) / total * 100, 2),
            "class_1_pct": round(counts.get(1, 0) / total * 100, 2),
        }
