from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def detect_column_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric and categorical lists."""
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    return numeric_columns, categorical_columns


def build_segmentation_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing for segmentation.

    Numeric features are imputed and standardized.
    Categorical features are imputed and one-hot encoded.
    """
    numeric_columns, categorical_columns = detect_column_types(features)
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype=np.float32,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No columns available to build segmentation preprocessor.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def transform_features_with_pca(
    features: pd.DataFrame,
    variance_threshold: float,
) -> tuple[np.ndarray, ColumnTransformer, PCA]:
    """Preprocess features and reduce dimensionality with PCA."""
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in the interval (0, 1].")

    preprocessor = build_segmentation_preprocessor(features)
    transformed = preprocessor.fit_transform(features)
    transformed_array = np.asarray(transformed, dtype=np.float32)

    pca = PCA(n_components=variance_threshold, svd_solver="full")
    transformed_pca = pca.fit_transform(transformed_array)
    return transformed_pca, preprocessor, pca


def _sample_indices(
    n_rows: int,
    max_rows: int | None,
    random_state: int,
) -> np.ndarray:
    """Return deterministic sample indices capped by max_rows."""
    if max_rows is None or n_rows <= max_rows:
        return np.arange(n_rows)

    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False))


def evaluate_kmeans_range(
    transformed_pca: np.ndarray,
    k_values: Iterable[int],
    random_state: int,
    fit_sample_size: int = 50_000,
    silhouette_sample_size: int = 20_000,
) -> pd.DataFrame:
    """
    Evaluate KMeans over candidate K values using inertia and silhouette score.

    To keep runtime practical on large datasets, fitting and silhouette are
    computed on deterministic samples when needed.
    """
    k_list = sorted(int(k) for k in k_values)
    if not k_list:
        raise ValueError("k_values cannot be empty.")
    if min(k_list) < 2:
        raise ValueError("All k_values must be >= 2.")

    fit_idx = _sample_indices(
        n_rows=transformed_pca.shape[0],
        max_rows=fit_sample_size,
        random_state=random_state,
    )
    fit_matrix = transformed_pca[fit_idx]

    rows: list[dict[str, float | int]] = []
    for k in k_list:
        kmeans = KMeans(
            n_clusters=k,
            n_init=20,
            max_iter=300,
            random_state=random_state,
        )
        fit_labels = kmeans.fit_predict(fit_matrix)

        silhouette_idx = _sample_indices(
            n_rows=fit_matrix.shape[0],
            max_rows=silhouette_sample_size,
            random_state=random_state + k,
        )
        silhouette_value = silhouette_score(
            fit_matrix[silhouette_idx],
            fit_labels[silhouette_idx],
            metric="euclidean",
        )

        rows.append(
            {
                "k": k,
                "inertia": float(kmeans.inertia_),
                "silhouette": float(silhouette_value),
                "fit_sample_size": int(fit_matrix.shape[0]),
                "silhouette_sample_size": int(silhouette_idx.shape[0]),
            }
        )

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def estimate_elbow_k(kmeans_metrics: pd.DataFrame) -> int:
    """Estimate elbow point from inertia curve using second differences."""
    ordered = kmeans_metrics.sort_values("k")
    k_values = ordered["k"].to_numpy()
    inertia_values = ordered["inertia"].to_numpy()

    if k_values.shape[0] < 3:
        return int(k_values[np.argmin(inertia_values)])

    second_diff = np.diff(inertia_values, n=2)
    elbow_idx = int(np.argmax(second_diff) + 1)
    return int(k_values[elbow_idx])


def select_reasonable_k(
    kmeans_metrics: pd.DataFrame,
    min_k: int = 3,
    max_k: int = 5,
    silhouette_tolerance: float = 0.01,
) -> int:
    """
    Select K constrained to [min_k, max_k] using silhouette + elbow evidence.

    Primary criterion is silhouette score; elbow estimate is used when
    performance is near-tied.
    """
    candidate_rows = kmeans_metrics[
        (kmeans_metrics["k"] >= min_k) & (kmeans_metrics["k"] <= max_k)
    ].copy()
    if candidate_rows.empty:
        raise ValueError(f"No candidate k found in [{min_k}, {max_k}].")

    best_row = candidate_rows.sort_values("silhouette", ascending=False).iloc[0]
    best_k = int(best_row["k"])
    best_silhouette = float(best_row["silhouette"])

    elbow_k = estimate_elbow_k(kmeans_metrics)
    if min_k <= elbow_k <= max_k:
        elbow_row = candidate_rows[candidate_rows["k"] == elbow_k].iloc[0]
        elbow_silhouette = float(elbow_row["silhouette"])
        if best_silhouette - elbow_silhouette <= silhouette_tolerance:
            return elbow_k

    return best_k


def fit_final_kmeans(
    transformed_pca: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> tuple[KMeans, np.ndarray]:
    """Fit final KMeans model on full PCA matrix."""
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=30,
        max_iter=400,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(transformed_pca)
    return kmeans, labels


def _weighted_category_shares(
    values: pd.Series,
    sample_weight: pd.Series,
) -> pd.Series:
    grouped = (
        pd.DataFrame(
            {
                "value": values.astype("string").fillna("Missing"),
                "weight": sample_weight.astype("float64"),
            }
        )
        .groupby("value", observed=False)["weight"]
        .sum()
    )
    total_weight = float(grouped.sum())
    if total_weight <= 0:
        return pd.Series(dtype="float64")
    return (grouped / total_weight).sort_values(ascending=False)


def format_top_weighted_categories(
    values: pd.Series,
    sample_weight: pd.Series,
    top_n: int = 3,
) -> str:
    """Format top-N weighted category shares as a compact string."""
    shares = _weighted_category_shares(values, sample_weight).head(top_n)
    if shares.empty:
        return "N/A"
    return "; ".join(f"{category} ({share * 100:.1f}%)" for category, share in shares.items())


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return float(np.sqrt(max(variance, 1e-12)))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1].")

    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumulative_weights = np.cumsum(sorted_weights)
    threshold = quantile * cumulative_weights[-1]
    quantile_idx = int(np.searchsorted(cumulative_weights, threshold))
    quantile_idx = min(max(quantile_idx, 0), sorted_values.shape[0] - 1)
    return float(sorted_values[quantile_idx])


def format_weighted_age_distribution(age_values: pd.Series, sample_weight: pd.Series) -> str:
    """Return weighted age-bin shares as a readable string."""
    valid_age = pd.to_numeric(age_values, errors="coerce")
    valid_mask = valid_age.notna()
    if not valid_mask.any():
        return "N/A"

    age = valid_age[valid_mask].astype("float64")
    weights = sample_weight[valid_mask].astype("float64")

    bins = [0, 18, 25, 35, 45, 55, 65, 120]
    labels = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    bucketed_age = pd.cut(age, bins=bins, labels=labels, right=False, include_lowest=True)
    shares = _weighted_category_shares(bucketed_age, weights)
    if shares.empty:
        return "N/A"
    return "; ".join(f"{label} ({share * 100:.1f}%)" for label, share in shares.items())


def _numeric_distinguishing_features(
    features: pd.DataFrame,
    cluster_labels: np.ndarray,
    sample_weight: pd.Series,
    top_n: int = 3,
) -> dict[int, str]:
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    if not numeric_columns:
        unique_clusters = np.unique(cluster_labels)
        return {int(cluster): "N/A" for cluster in unique_clusters}

    weights_all = sample_weight.to_numpy(dtype="float64")
    global_stats: dict[str, tuple[float, float, np.ndarray]] = {}
    for column in numeric_columns:
        values = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype="float64")
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() < 2:
            continue
        values_valid = values[valid_mask]
        weights_valid = weights_all[valid_mask]
        global_stats[column] = (
            _weighted_mean(values_valid, weights_valid),
            _weighted_std(values_valid, weights_valid),
            values,
        )

    cluster_signals: dict[int, str] = {}
    for cluster in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster
        cluster_weights_all = weights_all[cluster_mask]
        signals: list[tuple[float, str]] = []

        for column, (global_mean, global_std, column_values) in global_stats.items():
            cluster_values = column_values[cluster_mask]
            valid_cluster_mask = ~np.isnan(cluster_values)
            if valid_cluster_mask.sum() == 0:
                continue

            cluster_values_valid = cluster_values[valid_cluster_mask]
            cluster_weights_valid = cluster_weights_all[valid_cluster_mask]
            cluster_mean = _weighted_mean(cluster_values_valid, cluster_weights_valid)
            z_score = (cluster_mean - global_mean) / max(global_std, 1e-12)
            signals.append((abs(z_score), f"{column} ({z_score:+.2f} sd)"))

        signals.sort(key=lambda item: item[0], reverse=True)
        top_signals = [signal for _, signal in signals[:top_n]]
        cluster_signals[int(cluster)] = "; ".join(top_signals) if top_signals else "N/A"

    return cluster_signals


def _dominant_category(values: pd.Series, sample_weight: pd.Series) -> str:
    shares = _weighted_category_shares(values, sample_weight)
    if shares.empty:
        return "N/A"
    return str(shares.index[0])


def build_cluster_summary(
    features: pd.DataFrame,
    income_label: pd.Series,
    sample_weight: pd.Series,
    cluster_labels: np.ndarray,
    age_column: str = "age",
    education_column: str = "education",
    occupation_column: str = "major occupation code",
) -> pd.DataFrame:
    """Create weighted cluster-level profile table for business interpretation."""
    summary_rows: list[dict[str, float | int | str]] = []
    numeric_signals = _numeric_distinguishing_features(
        features=features,
        cluster_labels=cluster_labels,
        sample_weight=sample_weight,
        top_n=3,
    )

    unique_clusters = np.sort(np.unique(cluster_labels))
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_features = features.loc[cluster_mask]
        cluster_income = income_label.loc[cluster_mask].astype("float64")
        cluster_weights = sample_weight.loc[cluster_mask].astype("float64")
        cluster_total_weight = float(cluster_weights.sum())

        weighted_income_rate = float(
            np.average(cluster_income.to_numpy(dtype="float64"), weights=cluster_weights)
        )

        cluster_age = pd.to_numeric(cluster_features[age_column], errors="coerce")
        age_valid_mask = cluster_age.notna()
        if age_valid_mask.any():
            age_values = cluster_age[age_valid_mask].to_numpy(dtype="float64")
            age_weights = cluster_weights[age_valid_mask].to_numpy(dtype="float64")
            age_mean = _weighted_mean(age_values, age_weights)
            age_median = _weighted_quantile(age_values, age_weights, quantile=0.5)
        else:
            age_mean = np.nan
            age_median = np.nan

        education_profile = format_top_weighted_categories(
            cluster_features[education_column],
            cluster_weights,
            top_n=3,
        )
        occupation_profile = format_top_weighted_categories(
            cluster_features[occupation_column],
            cluster_weights,
            top_n=3,
        )

        dominant_education = _dominant_category(
            cluster_features[education_column], cluster_weights
        )
        dominant_occupation = _dominant_category(
            cluster_features[occupation_column], cluster_weights
        )

        summary_rows.append(
            {
                "cluster": int(cluster),
                "n_records": int(cluster_mask.sum()),
                "weighted_population": cluster_total_weight,
                "weighted_income_rate": weighted_income_rate,
                "age_mean": age_mean,
                "age_median": age_median,
                "age_distribution": format_weighted_age_distribution(
                    cluster_features[age_column], cluster_weights
                ),
                "education_distribution_top3": education_profile,
                "occupation_distribution_top3": occupation_profile,
                "key_distinguishing_features": (
                    f"{numeric_signals[int(cluster)]}; "
                    f"dominant_education={dominant_education}; "
                    f"dominant_occupation={dominant_occupation}"
                ),
            }
        )

    return pd.DataFrame(summary_rows).sort_values("cluster").reset_index(drop=True)


def create_segmentation_plots(
    kmeans_metrics: pd.DataFrame,
    transformed_pca: np.ndarray,
    cluster_labels: np.ndarray,
    final_kmeans: KMeans,
    explained_variance_ratio: np.ndarray,
    selected_k: int,
    output_path: str | Path,
    random_state: int,
    max_scatter_points: int = 20_000,
) -> None:
    """Create elbow/silhouette diagnostics and 2D PCA cluster projection plot."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ordered_metrics = kmeans_metrics.sort_values("k")
    k_values = ordered_metrics["k"].to_numpy(dtype=int)
    inertia_values = ordered_metrics["inertia"].to_numpy(dtype=float)
    silhouette_values = ordered_metrics["silhouette"].to_numpy(dtype=float)

    scatter_idx = _sample_indices(
        n_rows=transformed_pca.shape[0],
        max_rows=max_scatter_points,
        random_state=random_state,
    )
    scatter_matrix = transformed_pca[scatter_idx]
    scatter_labels = cluster_labels[scatter_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(k_values, inertia_values, marker="o", color="#1f77b4")
    axes[0].axvline(selected_k, linestyle="--", color="#d62728", label=f"Selected K={selected_k}")
    axes[0].set_title("Elbow Curve (Inertia)")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].legend()

    axes[1].plot(k_values, silhouette_values, marker="o", color="#2ca02c")
    axes[1].axvline(selected_k, linestyle="--", color="#d62728", label=f"Selected K={selected_k}")
    axes[1].set_title("Silhouette by K")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    scatter = axes[2].scatter(
        scatter_matrix[:, 0],
        scatter_matrix[:, 1],
        c=scatter_labels,
        s=8,
        alpha=0.6,
        cmap="tab10",
    )
    centers = final_kmeans.cluster_centers_
    if centers.shape[1] >= 2:
        axes[2].scatter(
            centers[:, 0],
            centers[:, 1],
            c="black",
            marker="X",
            s=180,
            label="Centroids",
        )
        axes[2].legend(loc="best")

    pc1_var = explained_variance_ratio[0] * 100 if explained_variance_ratio.size >= 1 else 0.0
    pc2_var = explained_variance_ratio[1] * 100 if explained_variance_ratio.size >= 2 else 0.0
    axes[2].set_title("PCA Projection of Clusters")
    axes[2].set_xlabel(f"PC1 ({pc1_var:.1f}% var)")
    axes[2].set_ylabel(f"PC2 ({pc2_var:.1f}% var)")
    fig.colorbar(scatter, ax=axes[2], fraction=0.046, pad=0.04, label="Cluster")

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
