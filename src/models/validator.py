"""
Validator — learning curves and cross-validation utilities.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from src.models.trainer import get_feature_cols, score


class Validator:
    """
    Computes learning curves by training at increasing data sizes.
    """

    def __init__(self, config):
        self.config = config
        self.eval_metric = config.get("eval_metric") or "f1"
        lc_config = config.get("learning_curves") or {}
        self.train_sizes = lc_config.get("train_sizes", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.enabled = lc_config.get("enabled", True)

    def compute_learning_curves(
        self,
        model_factory,
        train_df: pd.DataFrame,
        val_folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    ) -> dict:
        """
        Train model at each fraction of training data and evaluate on a
        fixed held-out validation set.

        The last val fold is used as the fixed evaluation set and is
        excluded from the training pool entirely — this prevents the
        progressive leakage that occurs when the model is evaluated on
        data it was trained on at larger training sizes.

        Args:
            model_factory: callable() -> unfitted sklearn model
            train_df: full training DataFrame
            val_folds: validation folds (last fold used as held-out eval set)

        Returns:
            dict with keys: train_sizes_abs, train_scores, val_scores, metric
        """
        feat_cols = get_feature_cols(train_df)

        # Hold out the last val fold as fixed evaluation set — never in training
        _, last_val_fold = val_folds[-1]
        train_pool = train_df[~train_df.index.isin(last_val_fold.index)]

        X_train_full = train_pool[feat_cols].values
        y_train_full = train_pool["target"].values
        X_val = last_val_fold[feat_cols].values
        y_val = last_val_fold["target"].values
        n = len(X_train_full)

        train_scores = []
        val_scores = []
        sizes_abs = []

        for frac in self.train_sizes:
            size = max(int(n * frac), 10)
            X_sub = X_train_full[:size]
            y_sub = y_train_full[:size]

            model = model_factory()
            model.fit(X_sub, y_sub)

            # Train score (in-sample)
            y_pred_train = model.predict(X_sub)
            y_prob_train = model.predict_proba(X_sub)[:, 1]
            train_score = score(y_sub, y_pred_train, y_prob_train, self.eval_metric)

            # Val score (always out-of-sample — last fold was never in training)
            y_pred_val = model.predict(X_val)
            y_prob_val = model.predict_proba(X_val)[:, 1]
            val_score = score(y_val, y_pred_val, y_prob_val, self.eval_metric)

            train_scores.append(train_score)
            val_scores.append(val_score)
            sizes_abs.append(size)

        return {
            "train_sizes": self.train_sizes,
            "train_sizes_abs": sizes_abs,
            "train_scores": train_scores,
            "val_scores": val_scores,
            "metric": self.eval_metric,
        }
