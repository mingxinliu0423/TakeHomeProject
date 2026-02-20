# Census Income Modeling for Retail Targeting

This repository contains two connected workflows on the 1994--95 Census Income data: weighted income classification and unsupervised segmentation for targeting strategy.  
The selected classifier is XGBoost, with hold-out ROC-AUC `0.9540`, weighted ROC-AUC `0.9540`, F1 `0.5820`, and Precision@20% `0.2844` (weighted `0.2921`).  
Validation base rates are `0.0620` (unweighted) and `0.0647` (weighted), so top-20% lift is `4.58x` unweighted and `4.52x` weighted.  
Logistic regression was still a strong baseline (ROC-AUC `0.9435`, Precision@20% `0.2768`), so XGBoost is a margin win rather than a huge jump.  
That pattern suggests a large share of predictive signal is captured by relatively low-order effects in the encoded feature space, with boosting adding incremental refinements.  
Segmentation used standardized numeric variables, one-hot encoded categoricals, PCA retaining `92.17%` variance, and KMeans with `K` constrained to `3..5`.  
`K=3` was retained for operational clarity, but silhouette scores are still low (`0.2270` at `K=3`, `0.2325` at `K=5`), so these clusters should be treated as coarse planning buckets, not sharply separated customer types.

## Repository Structure

```text
project_root/
|-- data/
|   |-- census-bureau.data
|   `-- census-bureau.columns
|-- src/
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- models.py
|   |-- metrics.py
|   |-- train.py
|   |-- clustering_utils.py
|   `-- segmentation.py
|-- artifacts/
|   |-- metrics.csv
|   |-- metrics_summary.csv
|   |-- best_model.pkl
|   |-- classification_roc_curves.png
|   |-- gains_table.csv
|   |-- gains_table_<model>.csv
|   |-- gains_curve.png
|   |-- calibration_curve.png
|   |-- cluster_summary.csv
|   |-- cluster_kmeans_metrics.csv
|   |-- segmentation_preprocessor.pkl
|   |-- segmentation_pca.pkl
|   |-- segmentation_kmeans.pkl
|   |-- segmentation_metadata.json
|   |-- segmentation_stability.json
|   |-- segment_messaging.csv
|   `-- segmentation_plots.png
|-- report/
|   `-- report.tex
|-- requirements.txt
`-- README.md
```

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Classification

```bash
python src/train.py
```

Outputs:
- `artifacts/metrics.csv`
- `artifacts/metrics_summary.csv`
- `artifacts/best_model.pkl`
- `artifacts/classification_roc_curves.png`
- `artifacts/gains_table.csv`
- `artifacts/gains_table_<model>.csv`
- `artifacts/gains_curve.png`
- `artifacts/calibration_curve.png`

## Run Segmentation

```bash
python src/segmentation.py
```

Outputs:
- `artifacts/cluster_summary.csv`
- `artifacts/cluster_kmeans_metrics.csv`
- `artifacts/segmentation_preprocessor.pkl`
- `artifacts/segmentation_pca.pkl`
- `artifacts/segmentation_kmeans.pkl`
- `artifacts/segmentation_metadata.json`
- `artifacts/segmentation_stability.json`
- `artifacts/segment_messaging.csv`
- `artifacts/segmentation_plots.png`

## Key Outputs

- `artifacts/metrics.csv`: primary seed-2026 model comparison (ROC-AUC, Precision@Top20, and Brier; each with weighted counterpart where applicable).
- `artifacts/metrics_summary.csv`: mean/std uncertainty summary across deterministic seeds `2026..2030`.
- `artifacts/gains_table.csv`: best-model gains table at 5/10/20/30% target depths.
- `artifacts/gains_table_<model>.csv`: same gains breakdown for each candidate classifier.
- `artifacts/gains_curve.png`: precision/lift by depth for the selected model.
- `artifacts/calibration_curve.png`: reliability view plus Brier scores for the selected model.
- `artifacts/cluster_summary.csv`: weighted cluster profiling table with demographics and work-intensity stats.
- `artifacts/segmentation_stability.json`: ARI-based stability check for final `K`.
- `artifacts/segment_messaging.csv`: compact cluster-to-message/channel hypothesis table.
- `artifacts/segmentation_preprocessor.pkl`, `artifacts/segmentation_pca.pkl`, `artifacts/segmentation_kmeans.pkl`: reproducible scoring path for assigning new records to clusters.

## Reproducibility Notes

Both workflows use fixed random seeds and schema-driven column loading (no hard-coded column indices).  
Sample weights are used during supervised training so the fitted classifier reflects survey sampling design rather than raw row counts.  
Primary comparison metrics remain sample-level ranking metrics (`roc_auc`, `f1`, `precision_at_top20`) because model ranking quality is evaluated on actual held-out records.  
`roc_auc_weighted` is also written to `artifacts/metrics.csv` as a population-aligned check; it should be read as a sensitivity metric, not a replacement for the ranking baseline.  
For segmentation, the fitted preprocessor, PCA, and KMeans objects are now serialized so new records can be assigned to clusters with the same transformation path used in development.

## Report

The full write-up is in `report/report.tex`.

Optional compile command (if LaTeX is installed):

```bash
cd report
pdflatex report.tex
```
