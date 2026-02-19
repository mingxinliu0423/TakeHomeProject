from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from clustering_utils import (
    build_cluster_summary,
    create_segmentation_plots,
    evaluate_kmeans_range,
    fit_final_kmeans,
    select_reasonable_k,
    transform_features_with_pca,
)
from data_loader import load_dataset

RANDOM_STATE = 42
PCA_VARIANCE_THRESHOLD = 0.92
K_VALUES = range(2, 9)
MIN_SELECTED_K = 3
MAX_SELECTED_K = 5
MAX_SCATTER_POINTS = 20_000


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "census-bureau.data"
    columns_path = project_root / "data" / "census-bureau.columns"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    features, income_label, sample_weight = load_dataset(
        data_path=data_path,
        columns_path=columns_path,
    )

    # Segmentation input excludes label and sample weight by design.
    transformed_pca, preprocessor, pca = transform_features_with_pca(
        features=features,
        variance_threshold=PCA_VARIANCE_THRESHOLD,
    )
    retained_variance = float(pca.explained_variance_ratio_.sum())

    kmeans_metrics = evaluate_kmeans_range(
        transformed_pca=transformed_pca,
        k_values=K_VALUES,
        random_state=RANDOM_STATE,
        fit_sample_size=50_000,
        silhouette_sample_size=20_000,
    )

    selected_k = select_reasonable_k(
        kmeans_metrics=kmeans_metrics,
        min_k=MIN_SELECTED_K,
        max_k=MAX_SELECTED_K,
        silhouette_tolerance=0.01,
    )

    final_kmeans, cluster_labels = fit_final_kmeans(
        transformed_pca=transformed_pca,
        n_clusters=selected_k,
        random_state=RANDOM_STATE,
    )

    preprocessor_path = artifacts_dir / "segmentation_preprocessor.pkl"
    pca_path = artifacts_dir / "segmentation_pca.pkl"
    kmeans_path = artifacts_dir / "segmentation_kmeans.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(pca, pca_path)
    joblib.dump(final_kmeans, kmeans_path)

    metadata_path = artifacts_dir / "segmentation_metadata.json"
    metadata = {
        "selected_k": int(selected_k),
        "pca_variance_threshold": float(PCA_VARIANCE_THRESHOLD),
        "retained_variance": retained_variance,
        "random_state": int(RANDOM_STATE),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "max_scatter_points": int(MAX_SCATTER_POINTS),
        "k_range_evaluated": list(K_VALUES),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    cluster_summary = build_cluster_summary(
        features=features,
        income_label=income_label,
        sample_weight=sample_weight,
        cluster_labels=cluster_labels,
    )

    cluster_summary_path = artifacts_dir / "cluster_summary.csv"
    cluster_summary.to_csv(cluster_summary_path, index=False)

    kmeans_metrics_path = artifacts_dir / "cluster_kmeans_metrics.csv"
    kmeans_metrics.to_csv(kmeans_metrics_path, index=False)

    plot_path = artifacts_dir / "segmentation_plots.png"
    create_segmentation_plots(
        kmeans_metrics=kmeans_metrics,
        transformed_pca=transformed_pca,
        cluster_labels=cluster_labels,
        final_kmeans=final_kmeans,
        explained_variance_ratio=pca.explained_variance_ratio_,
        selected_k=selected_k,
        output_path=plot_path,
        random_state=RANDOM_STATE,
        max_scatter_points=MAX_SCATTER_POINTS,
    )

    summary_preview = cluster_summary[
        ["cluster", "n_records", "weighted_income_rate", "age_mean"]
    ].copy()
    for column in ["weighted_income_rate", "age_mean"]:
        summary_preview[column] = summary_preview[column].map(lambda value: f"{value:.4f}")

    print("Segmentation completed")
    print(f"Retained PCA variance: {retained_variance:.4f}")
    print(f"Selected number of clusters (K): {selected_k}")
    print("\nKMeans diagnostics:")
    print(kmeans_metrics.to_string(index=False))
    print("\nCluster profile preview:")
    print(summary_preview.to_string(index=False))
    print(f"\nSaved cluster summary: {cluster_summary_path}")
    print(f"Saved KMeans diagnostics: {kmeans_metrics_path}")
    print(f"Saved segmentation plot: {plot_path}")
    print(f"Saved segmentation preprocessor: {preprocessor_path}")
    print(f"Saved segmentation PCA: {pca_path}")
    print(f"Saved segmentation KMeans: {kmeans_path}")
    print(f"Saved segmentation metadata: {metadata_path}")


if __name__ == "__main__":
    main()
