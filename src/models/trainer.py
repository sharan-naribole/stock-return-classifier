"""
Model trainer — Logistic Regression, Random Forest, XGBoost with HPT.
"""

import itertools
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier


FEATURE_COLS_EXCLUDE = ["target", "Adj_Close"]


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Get feature columns (exclude target and raw price cols)."""
    return [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]


def score(y_true, y_pred, y_prob, metric: str) -> float:
    """Compute a single evaluation metric."""
    if metric == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    elif metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "roc_auc":
        return roc_auc_score(y_true, y_prob)
    elif metric == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    elif metric == "recall":
        return recall_score(y_true, y_pred, zero_division=0)
    else:
        raise ValueError(f"Unknown metric: {metric}")


class ModelTrainer:
    """
    Trains and tunes models using validation folds.
    Supports: logistic_regression, random_forest, xgboost
    """

    def __init__(self, config):
        self.config = config
        self.eval_metric = config.get("eval_metric") or "f1"
        self.models_config = config.get("models") or {}

    def _build_model(self, model_name: str, params: dict):
        """Instantiate a model with given hyperparameters and class imbalance settings."""
        model_cfg = self.models_config.get(model_name, {})
        class_weight = model_cfg.get("class_weight", None)       # "balanced" or None
        scale_pos_weight = model_cfg.get("scale_pos_weight", 1)  # XGBoost imbalance weight

        if model_name == "logistic_regression":
            return LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 1000),
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
            )
        elif model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
            )
        elif model_name == "xgboost":
            return XGBClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 3),
                learning_rate=params.get("learning_rate", 0.1),
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _param_grid(self, model_name: str) -> List[dict]:
        """Generate all hyperparameter combinations for a model."""
        hparams = self.models_config.get(model_name, {}).get("hyperparameters", {})
        if not hparams:
            return [{}]

        # Separate list params from scalar params
        grid_params = {k: v for k, v in hparams.items() if isinstance(v, list)}
        fixed_params = {k: v for k, v in hparams.items() if not isinstance(v, list)}

        if not grid_params:
            return [fixed_params]

        keys = list(grid_params.keys())
        values = list(grid_params.values())
        combos = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            params.update(fixed_params)
            combos.append(params)
        return combos

    def _eval_on_folds(
        self,
        model_name: str,
        params: dict,
        val_folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    ) -> float:
        """Evaluate a model with given params across all val folds, return mean metric."""
        fold_scores = []
        for train_fold, val_fold in val_folds:
            feat_cols = get_feature_cols(train_fold)
            X_train = train_fold[feat_cols].values
            y_train = train_fold["target"].values
            X_val = val_fold[feat_cols].values
            y_val = val_fold["target"].values

            model = self._build_model(model_name, params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            fold_scores.append(score(y_val, y_pred, y_prob, self.eval_metric))

        return float(np.mean(fold_scores))

    def train_and_tune(
        self,
        train_df: pd.DataFrame,
        val_folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
        model_name: str,
    ) -> Tuple[object, dict, dict]:
        """
        HPT: evaluate all param combos on val folds, pick best.
        Returns: (best_model_fitted_on_full_train, best_params, val_metrics)
        """
        param_grid = self._param_grid(model_name)
        print(f"  {model_name}: testing {len(param_grid)} hyperparameter combinations...")

        best_score = -np.inf
        best_params = param_grid[0]

        for params in param_grid:
            avg_score = self._eval_on_folds(model_name, params, val_folds)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        print(f"  Best params: {best_params}, best {self.eval_metric}: {best_score:.4f}")

        # Re-fit on full training data
        feat_cols = get_feature_cols(train_df)
        X_train = train_df[feat_cols].values
        y_train = train_df["target"].values
        best_model = self._build_model(model_name, best_params)
        best_model.fit(X_train, y_train)

        val_metrics = {
            "best_val_" + self.eval_metric: best_score,
            "best_params": best_params,
        }

        return best_model, best_params, val_metrics

    def train_all(
        self,
        train_df: pd.DataFrame,
        val_folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    ) -> Tuple[dict, dict, dict]:
        """
        Train all enabled models.
        Returns: (models_dict, params_dict, metrics_dict)
        """
        models, params, metrics = {}, {}, {}
        for model_name, cfg in self.models_config.items():
            if cfg.get("enabled", True):
                print(f"\nTraining {model_name}...")
                model, best_params, val_metrics = self.train_and_tune(train_df, val_folds, model_name)
                models[model_name] = model
                params[model_name] = best_params
                metrics[model_name] = val_metrics
        return models, params, metrics

    def save_model(self, model, path: str):
        """Save model to pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Model saved to {path}")

    def load_model(self, path: str):
        """Load model from pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)
