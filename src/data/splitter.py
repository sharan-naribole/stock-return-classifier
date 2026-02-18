"""
Data splitter — temporal train/val/test splits with no lookahead bias.

Design:
  - train_df  : ALL pre-test data. Final model is trained on this after HPT.
  - val_folds : TimeSeriesSplit windows within train_df, used only for HPT scoring.
  - test_df   : held-out last test_years of data, never seen during training or HPT.
"""

from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class DataSplitter:
    """
    Splits data into train, validation folds, and test sets using temporal ordering.

    - test_df   : last test_years of data (held out entirely)
    - train_df  : everything before test — full dataset for final model fitting
    - val_folds : TimeSeriesSplit(n_folds) on train_df for HPT cross-validation
    """

    def __init__(self, config):
        self.config = config
        self.test_years = config.get("split.test_years") or 1
        self.validation_type = config.get("split.validation_type") or "expanding"
        self.n_folds = config.get("split.n_folds") or 5
        self.sliding_window_years = config.get("split.sliding_window_years") or 3

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
        """
        Returns (train_df, val_folds, test_df).

        train_df  : all pre-test data
        val_folds : list of (train_fold, val_fold) DataFrames for HPT
        test_df   : held-out test period
        """
        df = df.sort_index()

        # ── Test set: last test_years of data ─────────────────────────────────
        last_date = df.index[-1]
        test_start = last_date - pd.DateOffset(years=self.test_years)
        test_df = df[df.index > test_start]
        train_df = df[df.index <= test_start]

        if len(train_df) == 0:
            raise ValueError("No training data after reserving test set.")

        # ── Validation folds: CV splits within train_df ───────────────────────
        if self.validation_type == "expanding":
            val_folds = self._expanding_window(train_df)
        elif self.validation_type == "sliding":
            val_folds = self._sliding_window(train_df)
        else:
            raise ValueError(f"Unknown validation_type: {self.validation_type}")

        return train_df, val_folds, test_df

    def _expanding_window(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Expanding window CV using sklearn TimeSeriesSplit.
        Each fold's training set grows by one block; validation is the next block.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        folds = []
        for train_idx, val_idx in tscv.split(df):
            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]
            folds.append((train_fold, val_fold))
        return folds

    def _sliding_window(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Sliding window CV using sklearn TimeSeriesSplit with fixed max_train_size.
        """
        max_train_size = int(self.sliding_window_years * 252)
        tscv = TimeSeriesSplit(n_splits=self.n_folds, max_train_size=max_train_size)
        folds = []
        for train_idx, val_idx in tscv.split(df):
            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]
            folds.append((train_fold, val_fold))
        return folds

    def get_split_info(self, train_df, val_folds, test_df) -> dict:
        """Return summary info about the splits."""
        return {
            "train_start": str(train_df.index[0].date()),
            "train_end": str(train_df.index[-1].date()),
            "train_size": len(train_df),
            "n_val_folds": len(val_folds),
            "val_fold_sizes": [len(vf[1]) for vf in val_folds],
            "val_train_sizes": [len(vf[0]) for vf in val_folds],
            "test_start": str(test_df.index[0].date()),
            "test_end": str(test_df.index[-1].date()),
            "test_size": len(test_df),
        }
