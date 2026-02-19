from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import build_preprocessor


def build_logistic_regression_pipeline(
    train_features: pd.DataFrame, random_state: int
) -> Pipeline:
    preprocessor = build_preprocessor(train_features, scale_numeric=True)
    classifier = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def build_random_forest_pipeline(train_features: pd.DataFrame, random_state: int) -> Pipeline:
    preprocessor = build_preprocessor(train_features, scale_numeric=False)
    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def build_xgboost_pipeline(train_features: pd.DataFrame, random_state: int) -> Pipeline:
    preprocessor = build_preprocessor(train_features, scale_numeric=False)
    classifier = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def build_model_pipelines(train_features: pd.DataFrame, random_state: int) -> dict[str, Pipeline]:
    """Build all required model pipelines with integrated preprocessing."""
    return {
        "logistic_regression": build_logistic_regression_pipeline(
            train_features, random_state
        ),
        "random_forest": build_random_forest_pipeline(train_features, random_state),
        "xgboost": build_xgboost_pipeline(train_features, random_state),
    }
