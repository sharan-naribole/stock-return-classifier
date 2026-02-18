"""
Normalizer — rolling or standard Z-score normalization with no lookahead bias.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BINARY_COLS = ["Market_Trend", "target", "forward_return"]  # columns excluded from normalization


class Normalizer:
    """
    Two normalization strategies:
    
    - rolling (default): backward-only rolling Z-score applied on the full timeline
      (no lookahead because it only uses past values). Market_Trend excluded.
    - standard: fit StandardScaler on train, transform val/test.
    """

    def __init__(self, method: str = "rolling", window: int = 63):
        if method not in ("rolling", "standard"):
            raise ValueError(f"method must be 'rolling' or 'standard', got '{method}'")
        self.method = method
        self.window = window
        self.scaler = None
        self.feature_cols = None

    @classmethod
    def from_config(cls, config) -> "Normalizer":
        method = config.get("data.normalization") or "rolling"
        window = config.get("data.normalization_window") or 63
        return cls(method=method, window=window)

    def _get_cols_to_normalize(self, df: pd.DataFrame) -> List[str]:
        """Get numeric columns excluding binary/categorical ones."""
        return [c for c in df.select_dtypes(include=[np.number]).columns if c not in BINARY_COLS]

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        val_folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
        test_df: pd.DataFrame,
    ):
        """
        Normalize train, val_folds, and test data.
        
        Returns:
            (norm_train, norm_val_folds, norm_test)
        """
        if self.method == "rolling":
            return self._rolling_normalize(train_df, val_folds, test_df)
        else:
            return self._standard_normalize(train_df, val_folds, test_df)

    def _rolling_normalize(self, train_df, val_folds, test_df):
        """
        Rolling Z-score: normalize on the full continuous timeline.
        Uses only backward-looking window — no lookahead.
        """
        # Combine all data in temporal order to compute rolling stats
        all_val = pd.concat([vf[1] for vf in val_folds]) if val_folds else pd.DataFrame()
        all_data = pd.concat([train_df, all_val, test_df]).sort_index()
        all_data = all_data[~all_data.index.duplicated(keep="first")]

        cols = self._get_cols_to_normalize(all_data)
        self.feature_cols = cols

        rolling_mean = all_data[cols].rolling(window=self.window, min_periods=1).mean()
        rolling_std  = all_data[cols].rolling(window=self.window, min_periods=1).std().replace(0, 1e-8)
        normalized = all_data.copy()
        normalized[cols] = (all_data[cols] - rolling_mean) / rolling_std
        # std() returns NaN for the first row (undefined with 1 sample); fill with 0
        # (Z-score of 0 = at the mean — neutral imputation for the warm-up period)
        normalized[cols] = normalized[cols].fillna(0)

        # Split back
        norm_train = normalized.loc[train_df.index]
        norm_val_folds = []
        for train_fold, val_fold in val_folds:
            norm_train_fold = normalized.loc[train_fold.index]
            norm_val_fold = normalized.loc[val_fold.index]
            norm_val_folds.append((norm_train_fold, norm_val_fold))
        norm_test = normalized.loc[test_df.index]

        return norm_train, norm_val_folds, norm_test

    def _standard_normalize(self, train_df, val_folds, test_df):
        """
        Standard Z-score: fit on train, apply to val/test.
        """
        cols = self._get_cols_to_normalize(train_df)
        self.feature_cols = cols
        self.scaler = StandardScaler()
        self.scaler.fit(train_df[cols])

        norm_train = train_df.copy()
        norm_train[cols] = self.scaler.transform(train_df[cols])

        norm_val_folds = []
        for train_fold, val_fold in val_folds:
            norm_train_fold = train_fold.copy()
            norm_train_fold[cols] = self.scaler.transform(train_fold[cols])
            norm_val_fold = val_fold.copy()
            norm_val_fold[cols] = self.scaler.transform(val_fold[cols])
            norm_val_folds.append((norm_train_fold, norm_val_fold))

        norm_test = test_df.copy()
        norm_test[cols] = self.scaler.transform(test_df[cols])

        return norm_train, norm_val_folds, norm_test
