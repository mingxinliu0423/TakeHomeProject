from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from metrics import evaluate_predictions_with_optional_weights
from models import build_model_pipelines

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 2026
VALIDATION_SIZE = 0.2
TOP_K = 0.2
GAINS_DEPTHS = [0.05, 0.10, 0.20, 0.30]
UNCERTAINTY_SEEDS = [2026, 2027, 2028, 2029, 2030]

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


def build_gains_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: np.ndarray,
    depths: list[float],
) -> pd.DataFrame:
    """Build gains table at fixed depth cutoffs for targeting decisions."""
    y_true_arr = np.asarray(y_true).astype(float)
    y_score_arr = np.asarray(y_score).astype(float)
    sample_weight_arr = np.asarray(sample_weight).astype(float)

    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must have the same number of samples.")
    if y_true_arr.shape[0] != sample_weight_arr.shape[0]:
        raise ValueError("sample_weight must have the same number of samples as y_true.")

    n_samples = y_true_arr.shape[0]
    ranking = np.argsort(-y_score_arr)
    base_rate_unweighted = float(np.mean(y_true_arr))
    base_rate_weighted = float(np.average(y_true_arr, weights=sample_weight_arr))

    rows: list[dict[str, float | int]] = []
    for depth in depths:
        if not 0 < depth <= 1:
            raise ValueError("Depth values must be in (0, 1].")

        n_selected = max(1, int(np.ceil(n_samples * depth)))
        top_idx = ranking[:n_selected]

        selected_y = y_true_arr[top_idx]
        selected_w = sample_weight_arr[top_idx]
        selected_weight_sum = float(np.sum(selected_w))

        precision_unweighted = float(np.mean(selected_y))
        precision_weighted = (
            float(np.sum(selected_w * selected_y) / selected_weight_sum)
            if selected_weight_sum > 0
            else float("nan")
        )

        lift_unweighted = (
            float(precision_unweighted / base_rate_unweighted)
            if base_rate_unweighted > 0
            else float("nan")
        )
        lift_weighted = (
            float(precision_weighted / base_rate_weighted)
            if base_rate_weighted > 0 and not np.isnan(precision_weighted)
            else float("nan")
        )

        rows.append(
            {
                "depth": float(depth),
                "selected_count": int(n_selected),
                "selected_weight_sum": selected_weight_sum,
                "precision_unweighted": precision_unweighted,
                "lift_unweighted": lift_unweighted,
                "precision_weighted": precision_weighted,
                "lift_weighted": lift_weighted,
                "base_rate_unweighted": base_rate_unweighted,
                "base_rate_weighted": base_rate_weighted,
            }
        )

    return pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)


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


def save_gains_curve(
    gains_df: pd.DataFrame,
    output_path: Path,
    model_name: str,
) -> None:
    """Plot precision and lift vs depth for the selected model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax_precision = plt.subplots(figsize=(8, 5))
    depth_pct = gains_df["depth"].to_numpy(dtype=float) * 100

    ax_precision.plot(
        depth_pct,
        gains_df["precision_unweighted"].to_numpy(dtype=float),
        marker="o",
        linewidth=2,
        label="Precision (unweighted)",
    )
    ax_precision.plot(
        depth_pct,
        gains_df["precision_weighted"].to_numpy(dtype=float),
        marker="o",
        linewidth=2,
        label="Precision (weighted)",
    )
    ax_precision.set_xlabel("Target Depth (%)")
    ax_precision.set_ylabel("Precision")
    ax_precision.grid(alpha=0.25, linestyle=":")

    ax_lift = ax_precision.twinx()
    ax_lift.plot(
        depth_pct,
        gains_df["lift_unweighted"].to_numpy(dtype=float),
        linestyle="--",
        linewidth=2,
        label="Lift (unweighted)",
        color="#2ca02c",
    )
    ax_lift.plot(
        depth_pct,
        gains_df["lift_weighted"].to_numpy(dtype=float),
        linestyle="--",
        linewidth=2,
        label="Lift (weighted)",
        color="#d62728",
    )
    ax_lift.set_ylabel("Lift")

    handles_1, labels_1 = ax_precision.get_legend_handles_labels()
    handles_2, labels_2 = ax_lift.get_legend_handles_labels()
    ax_precision.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")
    ax_precision.set_title(f"Gains Curve ({model_name})")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_calibration_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    brier: float,
    brier_weighted: float,
    output_path: Path,
    model_name: str,
) -> None:
    """Save a lightweight reliability plot for the selected model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    frac_pos, mean_pred = calibration_curve(
        y_true_arr,
        y_score_arr,
        n_bins=10,
        strategy="quantile",
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, label="Observed")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Positive Rate")
    ax.set_title(
        f"Calibration Curve ({model_name})\nBrier={brier:.4f}, Brier(w)={brier_weighted:.4f}"
    )
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_uncertainty_evaluation(
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series,
    seeds: list[int],
    top_k: float,
) -> pd.DataFrame:
    """
    Repeat split-train-evaluate over deterministic seeds and summarize uncertainty.

    We keep this pretty small on purpose, five splits is enough for a quick variance check.
    """
    repeat_rows: list[dict[str, float | int | str]] = []
    for split_seed in seeds:
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            features,
            target,
            sample_weight,
            test_size=VALIDATION_SIZE,
            random_state=split_seed,
            stratify=target,
        )

        models = build_model_pipelines(train_features=X_train, random_state=split_seed)
        base_rate_unweighted = float(y_val.mean())
        base_rate_weighted = float(np.average(y_val.values, weights=w_val.values))

        for model_name, model_pipeline in models.items():
            model_pipeline.fit(X_train, y_train, classifier__sample_weight=w_train)
            y_prob = model_pipeline.predict_proba(X_val)[:, 1]
            model_metrics = evaluate_predictions_with_optional_weights(
                y_true=y_val.values,
                y_score=y_prob,
                sample_weight=w_val.values,
                top_k=top_k,
            )

            p20 = float(model_metrics["precision_at_top20"])
            p20_weighted = float(model_metrics["precision_at_top20_weighted"])
            lift20 = p20 / base_rate_unweighted if base_rate_unweighted > 0 else float("nan")
            lift20_weighted = (
                p20_weighted / base_rate_weighted
                if base_rate_weighted > 0 and not np.isnan(p20_weighted)
                else float("nan")
            )

            repeat_rows.append(
                {
                    "split_seed": int(split_seed),
                    "model": model_name,
                    "roc_auc": float(model_metrics["roc_auc"]),
                    "roc_auc_weighted": float(model_metrics["roc_auc_weighted"]),
                    "precision_at_top20": p20,
                    "precision_at_top20_weighted": p20_weighted,
                    "lift_at_top20": float(lift20),
                    "lift_at_top20_weighted": float(lift20_weighted),
                }
            )

    repeat_df = pd.DataFrame(repeat_rows)
    summary_metrics = [
        "roc_auc",
        "roc_auc_weighted",
        "precision_at_top20",
        "precision_at_top20_weighted",
        "lift_at_top20",
        "lift_at_top20_weighted",
    ]

    summary_rows: list[dict[str, float | str]] = []
    for model_name, group in repeat_df.groupby("model", sort=False):
        row: dict[str, float | str] = {"model": model_name}
        for metric_name in summary_metrics:
            values = group[metric_name].astype(float)
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_std"] = float(values.std(ddof=0))
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)


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
    gains_tables: dict[str, pd.DataFrame] = {}
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
        # this table is for PM folks, keeps depth math obvious.
        gains_table = build_gains_table(
            y_true=y_val.values,
            y_score=y_prob,
            sample_weight=w_val.values,
            depths=GAINS_DEPTHS,
        )

        metric_rows.append({"model": model_name, **model_metrics})
        fitted_models[model_name] = model_pipeline
        validation_probabilities[model_name] = y_prob
        gains_tables[model_name] = gains_table

        per_model_gains_path = artifacts_dir / f"gains_table_{model_name}.csv"
        gains_table.to_csv(per_model_gains_path, index=False)

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

    best_model_gains_path = artifacts_dir / "gains_table.csv"
    best_model_gains = gains_tables[best_model_name].copy()
    best_model_gains["model"] = best_model_name
    best_model_gains.to_csv(best_model_gains_path, index=False)

    gains_curve_path = artifacts_dir / "gains_curve.png"
    save_gains_curve(
        gains_df=gains_tables[best_model_name],
        output_path=gains_curve_path,
        model_name=best_model_name,
    )

    best_model_row = metrics_df[metrics_df["model"] == best_model_name].iloc[0]
    calibration_curve_path = artifacts_dir / "calibration_curve.png"
    save_calibration_curve(
        y_true=y_val.values,
        y_score=validation_probabilities[best_model_name],
        brier=float(best_model_row["brier"]),
        brier_weighted=float(best_model_row["brier_weighted"]),
        output_path=calibration_curve_path,
        model_name=best_model_name,
    )

    metrics_summary_df = run_uncertainty_evaluation(
        features=X,
        target=y,
        sample_weight=sample_weight,
        seeds=UNCERTAINTY_SEEDS,
        top_k=TOP_K,
    )
    metrics_summary_path = artifacts_dir / "metrics_summary.csv"
    metrics_summary_df.to_csv(metrics_summary_path, index=False)

    pretty_metrics = metrics_df.copy()
    for metric_column in [
        "roc_auc",
        "roc_auc_weighted",
        "f1",
        "precision_at_top20",
        "precision_at_top20_weighted",
        "brier",
        "brier_weighted",
    ]:
        pretty_metrics[metric_column] = pretty_metrics[metric_column].map(lambda value: f"{value:.4f}")

    print("Validation Performance Comparison")
    print(pretty_metrics.to_string(index=False))
    print(f"\nSelected best model: {best_model_name}")
    print(f"Saved metrics: {metrics_output_path}")
    print(f"Saved ROC curves: {roc_plot_path}")
    print(f"Saved best model: {best_model_path}")
    print(f"Saved best-model gains table: {best_model_gains_path}")
    print(f"Saved gains curve: {gains_curve_path}")
    print(f"Saved calibration curve: {calibration_curve_path}")
    print(f"Saved uncertainty summary: {metrics_summary_path}")


if __name__ == "__main__":
    main()
