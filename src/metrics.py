from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.2) -> float:
    """Compute precision among the top-k fraction ranked by predicted score."""
    if not 0 < k <= 1:
        raise ValueError("k must be in the interval (0, 1].")

    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must have the same number of samples.")

    n_selected = max(1, int(np.ceil(y_score_arr.shape[0] * k)))
    top_indices = np.argsort(-y_score_arr)[:n_selected]
    return float(np.mean(y_true_arr[top_indices]))


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    top_k: float = 0.2,
) -> dict[str, float]:
    """Compute required classification metrics from probabilities."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    y_pred = (y_score_arr >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "precision_at_top20": float(precision_at_k(y_true_arr, y_score_arr, k=top_k)),
    }


def evaluate_predictions_with_optional_weights(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: np.ndarray | None = None,
    threshold: float = 0.5,
    top_k: float = 0.2,
) -> dict[str, float]:
    """
    Compute classification metrics and optionally include weighted ROC-AUC.

    ROC-AUC and ranking metrics remain sample-level by default; weighted ROC-AUC
    is added as a population-aligned diagnostic when sample_weight is provided.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must have the same number of samples.")

    y_pred = (y_score_arr >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "roc_auc_weighted": float("nan"),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "precision_at_top20": float(precision_at_k(y_true_arr, y_score_arr, k=top_k)),
    }

    if sample_weight is not None:
        sample_weight_arr = np.asarray(sample_weight)
        if sample_weight_arr.shape[0] != y_true_arr.shape[0]:
            raise ValueError("sample_weight must have the same number of samples as y_true.")
        metrics["roc_auc_weighted"] = float(
            roc_auc_score(y_true_arr, y_score_arr, sample_weight=sample_weight_arr)
        )

    return metrics
