"""
Classification metrics and evaluation plots.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ClassificationMetrics:
    """
    Computes classification metrics and generates evaluation plots.
    """

    def compute(self, y_true, y_pred, y_prob) -> Dict[str, float]:
        """
        Compute standard classification metrics.
        
        Returns dict: precision, recall, f1, accuracy, roc_auc
        """
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        }

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot normalized confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        axes[0].set_title(f"{title} (Counts)")
        axes[0].set_ylabel("Actual")
        axes[0].set_xlabel("Predicted")

        # Normalized
        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=axes[1],
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        axes[1].set_title(f"{title} (Normalized)")
        axes[1].set_ylabel("Actual")
        axes[1].set_xlabel("Predicted")

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        return fig

    def plot_roc_curve(
        self,
        y_true,
        y_prob,
        title: str = "ROC Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="royalblue", lw=2, label=f"ROC (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        return fig

    def plot_calibration_curve(
        self,
        y_true,
        y_prob,
        title: str = "Calibration Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot calibration (reliability) curve."""
        fraction_of_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Calibration curve
        axes[0].plot(mean_predicted, fraction_of_positives, "s-", label="Model", color="royalblue")
        axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfectly Calibrated")
        axes[0].set_xlabel("Mean Predicted Probability")
        axes[0].set_ylabel("Fraction of Positives")
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Probability histogram
        axes[1].hist(y_prob, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
        axes[1].set_xlabel("Predicted Probability")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Predicted Probability Distribution")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        return fig
