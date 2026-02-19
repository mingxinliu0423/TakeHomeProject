from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_column_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Detect numeric and categorical feature columns from dataframe dtypes."""
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    return numeric_columns, categorical_columns


def build_preprocessor(features: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    """
    Build a ColumnTransformer for mixed-type tabular data.

    Numeric columns use median imputation and optional standardization.
    Categorical columns use most-frequent imputation + one-hot encoding.
    """
    numeric_columns, categorical_columns = detect_column_types(features)
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        numeric_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy="median"))
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps=numeric_steps), numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No feature columns detected for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")
