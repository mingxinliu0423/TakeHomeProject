# Census Income Modeling for Retail Targeting

This repository contains two connected workflows on the 1994--95 Census Income data: weighted income classification and unsupervised segmentation for targeting strategy.  
The selected classifier is XGBoost, with hold-out ROC-AUC `0.9552`, F1 `0.5871`, and Precision@20% `0.2853`.  
The validation base rate is `0.0620`, so Precision@20% corresponds to roughly `4.60x` lift at the top of the ranked list.  
Logistic regression was a stronger baseline than expected (ROC-AUC `0.9449`, Precision@20% `0.2800`), and the gap to XGBoost is modest rather than structural.  
That pattern suggests a large share of predictive signal is captured by relatively low-order effects in the encoded feature space, with boosting adding incremental refinements.  
Segmentation used standardized numeric variables, one-hot encoded categoricals, PCA retaining `92.17%` variance, and KMeans with `K` constrained to `3..5`.  
`K=3` was retained for operational clarity, but silhouette scores are low (`0.2257` at `K=3`, `0.2310` at `K=5`), so these clusters should be treated as coarse planning buckets, not sharply separated customer types.

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
|   |-- best_model.pkl
|   |-- classification_roc_curves.png
|   |-- cluster_summary.csv
|   |-- cluster_kmeans_metrics.csv
|   |-- segmentation_preprocessor.pkl
|   |-- segmentation_pca.pkl
|   |-- segmentation_kmeans.pkl
|   |-- segmentation_metadata.json
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
- `artifacts/best_model.pkl`
- `artifacts/classification_roc_curves.png`

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
- `artifacts/segmentation_plots.png`

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
