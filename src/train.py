from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from metrics import evaluate_predictions_with_optional_weights
from models import build_model_pipelines

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
TOP_K = 0.2

MODEL_SIMPLICITY_RANK = {
    "logistic_regression": 1,
    "random_forest": 2,
    "xgboost": 3,
}


def select_best_model(metrics_df: pd.DataFrame) -> str:
    """
    Select model using performance first, then simplicity as a tradeoff.

    Candidates that are near-best on ROC-AUC and Precision@Top20 are retained,
    then the simpler model wins among those candidates.
    """
    working_df = metrics_df.copy()
    working_df["simplicity_rank"] = working_df["model"].map(MODEL_SIMPLICITY_RANK)

    best_roc_auc = working_df["roc_auc"].max()
    best_precision_top20 = working_df["precision_at_top20"].max()

    near_best = working_df[
        (working_df["roc_auc"] >= best_roc_auc - 0.002)
        & (working_df["precision_at_top20"] >= best_precision_top20 - 0.01)
    ]
    if near_best.empty:
        near_best = working_df

    selected_row = near_best.sort_values(
        by=["roc_auc", "precision_at_top20", "simplicity_rank"],
        ascending=[False, False, True],
    ).iloc[0]
    return str(selected_row["model"])


def save_roc_curve_plot(
    y_true: pd.Series,
    model_probabilities: dict[str, pd.Series],
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Persist multi-model ROC curve comparison for report-ready visualization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    auc_by_model = metrics_df.set_index("model")["roc_auc"].to_dict()

    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, probabilities in model_probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        auc_value = auc_by_model.get(model_name, float("nan"))
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_value:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Chance")
    ax.set_title("Validation ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "census-bureau.data"
    columns_path = project_root / "data" / "census-bureau.columns"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X, y, sample_weight = load_dataset(data_path=data_path, columns_path=columns_path)

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X,
        y,
        sample_weight,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_model_pipelines(train_features=X_train, random_state=RANDOM_STATE)
    fitted_models: dict[str, object] = {}
    validation_probabilities: dict[str, pd.Series] = {}
    metric_rows: list[dict[str, float | str]] = []

    for model_name, model_pipeline in models.items():
        model_pipeline.fit(X_train, y_train, classifier__sample_weight=w_train)
        y_prob = model_pipeline.predict_proba(X_val)[:, 1]
        model_metrics = evaluate_predictions_with_optional_weights(
            y_true=y_val.values,
            y_score=y_prob,
            sample_weight=w_val.values,
            top_k=TOP_K,
        )

        metric_rows.append({"model": model_name, **model_metrics})
        fitted_models[model_name] = model_pipeline
        validation_probabilities[model_name] = y_prob

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        by=["roc_auc", "precision_at_top20"], ascending=[False, False]
    )

    metrics_output_path = artifacts_dir / "metrics.csv"
    metrics_df.to_csv(metrics_output_path, index=False)

    roc_plot_path = artifacts_dir / "classification_roc_curves.png"
    save_roc_curve_plot(
        y_true=y_val,
        model_probabilities=validation_probabilities,
        metrics_df=metrics_df,
        output_path=roc_plot_path,
    )

    best_model_name = select_best_model(metrics_df)
    best_model_path = artifacts_dir / "best_model.pkl"
    joblib.dump(fitted_models[best_model_name], best_model_path)

    pretty_metrics = metrics_df.copy()
    for metric_column in ["roc_auc", "roc_auc_weighted", "f1", "precision_at_top20"]:
        pretty_metrics[metric_column] = pretty_metrics[metric_column].map(lambda value: f"{value:.4f}")

    print("Validation Performance Comparison")
    print(pretty_metrics.to_string(index=False))
    print(f"\nSelected best model: {best_model_name}")
    print(f"Saved metrics: {metrics_output_path}")
    print(f"Saved ROC curves: {roc_plot_path}")
    print(f"Saved best model: {best_model_path}")


if __name__ == "__main__":
    main()
