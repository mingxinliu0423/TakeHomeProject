from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_LABEL_COLUMN = "label"
DEFAULT_WEIGHT_COLUMN = "weight"


def load_column_names(columns_path: str | Path) -> list[str]:
    """Load column names from the provided schema file."""
    path = Path(columns_path)
    with path.open("r", encoding="utf-8") as file:
        columns = [line.strip() for line in file if line.strip()]

    if not columns:
        raise ValueError(f"No column names found in {path}.")
    return columns


def load_dataset(
    data_path: str | Path,
    columns_path: str | Path,
    label_column: str = DEFAULT_LABEL_COLUMN,
    weight_column: str = DEFAULT_WEIGHT_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load dataset and return features, binary label, and sample weights.

    The dataset uses '?' to represent missing values and a textual income label.
    """
    column_names = load_column_names(columns_path)
    dataset = pd.read_csv(
        data_path,
        header=None,
        names=column_names,
        na_values="?",
        skipinitialspace=True,
    )

    missing_required = {label_column, weight_column}.difference(dataset.columns)
    if missing_required:
        missing_str = ", ".join(sorted(missing_required))
        raise ValueError(f"Missing required columns: {missing_str}")

    normalized_label = (
        dataset[label_column]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
    )
    observed_values = set(normalized_label.dropna().unique())
    expected_values = {"-50000.", "50000+."}
    unexpected_values = observed_values.difference(expected_values)
    if unexpected_values:
        values = ", ".join(sorted(unexpected_values))
        raise ValueError(f"Unexpected label values found: {values}")

    y = normalized_label.eq("50000+.").astype("int8").rename(label_column)

    sample_weight = pd.to_numeric(dataset[weight_column], errors="coerce")
    if sample_weight.isna().any():
        missing_count = int(sample_weight.isna().sum())
        raise ValueError(
            f"Sample weight column '{weight_column}' has {missing_count} invalid rows."
        )
    sample_weight = sample_weight.astype("float64").rename(weight_column)

    X = dataset.drop(columns=[label_column, weight_column])
    return X, y, sample_weight
