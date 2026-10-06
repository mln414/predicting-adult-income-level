"""Training and input preparation for the interactive Adult Income demo."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "adult.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "results" / "outputs" / "adult_processed.csv"
SAVED_MODELS_DIR = PROJECT_ROOT / "results" / "saved_models"

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]
CATEGORICAL_FEATURES = [
    "workclass",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]
TARGET_COLUMN = "income"

AVAILABLE_MODELS = {
    "K-Nearest Neighbors (KNN)": "IT25102064",
    "Logistic Regression": "IT25102219",
    "Decision Tree": "IT25101220",
    "Random Forest": "IT25300345",
    "Gradient Boosting": "IT25300115",
    "Support Vector Machine (SVM)": "IT25103014",
}
UNAVAILABLE_MODELS = {}


class SVMProbabilityWrapper:
    """Wrapper around SVC providing probability estimates via calibrated sigmoid transformation."""

    def __init__(self, svc: SVC):
        self.svc = svc
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.svc.fit(X, y)
        self.classes_ = np.array(self.svc.classes_)
        return self

    def predict(self, X):
        return self.svc.predict(X)

    def predict_proba(self, X):
        decision = self.svc.decision_function(X)
        prob_1 = 1.0 / (1.0 + np.exp(-np.clip(decision, -500, 500)))
        return np.column_stack([1.0 - prob_1, prob_1])


def _prepare_project_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, StandardScaler, dict[str, list[str]]]:
    """Recreate the shared preprocessing and verify it matches the prepared CSV."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA_PATH}")
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {PROCESSED_DATA_PATH}. "
            "Run group_pipeline.ipynb first."
        )

    raw = pd.read_csv(RAW_DATA_PATH)
    for column in raw.select_dtypes(include=["object", "string"]).columns:
        raw[column] = raw[column].astype(str).str.strip()
    raw.replace("?", np.nan, inplace=True)

    for column in ["workclass", "occupation", "native.country"]:
        raw[column] = raw[column].fillna(raw[column].mode()[0])

    raw.drop_duplicates(inplace=True)
    raw.reset_index(drop=True, inplace=True)

    outlier_columns = ["capital.gain", "capital.loss", "hours.per.week"]
    outlier_mask = pd.Series(False, index=raw.index)
    for column in outlier_columns:
        outlier_mask |= raw[column] > raw[column].quantile(0.995)
    raw = raw.loc[~outlier_mask].copy().reset_index(drop=True)

    raw.drop(columns=["education"], inplace=True)
    user_input_reference = raw.drop(columns=[TARGET_COLUMN]).copy()
    category_levels = {
        column: sorted(raw[column].dropna().unique().tolist())
        for column in CATEGORICAL_FEATURES
    }
    raw[TARGET_COLUMN] = raw[TARGET_COLUMN].map({"<=50K": 0, ">50K": 1})
    if raw[TARGET_COLUMN].isna().any():
        raise ValueError("The raw dataset contains an unrecognized income label.")

    raw = pd.get_dummies(
        raw,
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=int,
    )
    scaler = StandardScaler()
    raw[NUMERIC_FEATURES] = scaler.fit_transform(raw[NUMERIC_FEATURES])

    expected = pd.read_csv(PROCESSED_DATA_PATH)
    if list(raw.columns) != list(expected.columns) or raw.shape != expected.shape:
        raise ValueError(
            "The raw dataset preprocessing does not match adult_processed.csv. "
            "Regenerate the prepared dataset by running group_pipeline.ipynb."
        )
    if not np.allclose(
        raw.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=1e-7,
        atol=1e-8,
    ):
        raise ValueError(
            "The raw and prepared datasets do not match. "
            "Regenerate adult_processed.csv by running group_pipeline.ipynb."
        )

    feature_columns = [column for column in expected.columns if column != TARGET_COLUMN]
    X = expected[feature_columns].astype(float)
    y = expected[TARGET_COLUMN].astype(int)
    return X, y, user_input_reference, scaler, category_levels


def load_demo_bundle() -> dict[str, Any]:
    """Train or load all 6 group models and metadata for the interactive demo."""
    X, y, input_reference, input_scaler, category_levels = _prepare_project_data()

    model_definitions: dict[str, Any] = {
        "K-Nearest Neighbors (KNN)": Pipeline(
            [
                ("scaler", RobustScaler()),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=15,
                        weights="distance",
                        metric="manhattan",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Logistic Regression": LogisticRegression(
            C=100,
            class_weight="balanced",
            penalty="l2",
            solver="lbfgs",
            max_iter=3000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            criterion="gini",
            max_depth=10,
            min_samples_leaf=5,
            min_samples_split=20,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_split=10,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            subsample=0.8,
            random_state=42,
        ),
        "Support Vector Machine (SVM)": SVMProbabilityWrapper(
            SVC(
                kernel="rbf",
                C=1,
                gamma="scale",
                class_weight="balanced",
                random_state=42,
            )
        ),
    }

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}

    for name, model in model_definitions.items():
        member_id = AVAILABLE_MODELS.get(name, "model")
        slug = f"{member_id}_{name.replace(' ', '_').replace('(', '').replace(')', '')}"
        saved_file = SAVED_MODELS_DIR / f"{slug}.joblib"

        if saved_file.exists():
            try:
                models[name] = joblib.load(saved_file)
                continue
            except Exception:
                pass

        model.fit(X, y)
        try:
            joblib.dump(model, saved_file)
        except Exception:
            pass
        models[name] = model

    numeric_defaults = {
        column: {
            "min": float(input_reference[column].min()),
            "max": float(input_reference[column].max()),
            "value": float(input_reference[column].median()),
        }
        for column in NUMERIC_FEATURES
    }
    categorical_defaults = {
        column: input_reference[column].mode()[0]
        for column in CATEGORICAL_FEATURES
    }

    return {
        "models": models,
        "feature_columns": list(X.columns),
        "input_scaler": input_scaler,
        "category_levels": category_levels,
        "numeric_defaults": numeric_defaults,
        "categorical_defaults": categorical_defaults,
        "training_rows": len(X),
    }


def transform_user_input(user_input: dict[str, Any], bundle: dict[str, Any]) -> pd.DataFrame:
    """Convert original Adult Income fields into the trained models' feature format."""
    feature_columns = bundle["feature_columns"]
    transformed = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    numeric_values = pd.DataFrame(
        [[user_input[column] for column in NUMERIC_FEATURES]],
        columns=NUMERIC_FEATURES,
    )
    scaled_values = bundle["input_scaler"].transform(numeric_values)[0]
    for column, value in zip(NUMERIC_FEATURES, scaled_values):
        transformed.at[0, column] = value

    for column in CATEGORICAL_FEATURES:
        value = user_input[column]
        levels = bundle["category_levels"][column]
        if value not in levels:
            raise ValueError(f"Unsupported value {value!r} for {column}.")
        if value != levels[0]:
            dummy_column = f"{column}_{value}"
            if dummy_column not in transformed.columns:
                raise ValueError(
                    f"Encoded feature {dummy_column!r} is not present in the "
                    "prepared dataset."
                )
            transformed.at[0, dummy_column] = 1.0

    return transformed
