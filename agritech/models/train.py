import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agritech.config import ARTIFACTS_DATA_DIR, ARTIFACTS_MODELS_DIR, MLRUNS_DIR
from agritech.data.preprocess import save_master_dataset

FEATURE_COLUMNS = [
    "Area",
    "Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]
TARGET_COLUMN = "hg_ha_yield"


def load_training_dataset(data_path: Path | None = None) -> pd.DataFrame:
    if data_path is None:
        data_path = ARTIFACTS_DATA_DIR / "consolidated_yield.csv"
    if not data_path.exists():
        save_master_dataset()
    return pd.read_csv(data_path)


def build_pipeline(model: object) -> Pipeline:
    categorical_features = ["Area", "Item"]
    numeric_features = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate_predictions(y_true: pd.Series, predictions: list[float]) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def train_and_log_models(data_path: Path | None = None) -> tuple[Pipeline, dict]:
    dataset = load_training_dataset(data_path)
    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("agritech-yield")

    best_model = None
    best_metrics = None
    best_rmse = float("inf")

    for name, model in candidates.items():
        pipeline = build_pipeline(model)
        with mlflow.start_run(run_name=name):
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            metrics = evaluate_predictions(y_test, predictions)

            mlflow.log_param("model_name", name)
            mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("r2", metrics["r2"])

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = pipeline
                best_metrics = metrics | {"model_name": name}

    if best_model is None or best_metrics is None:
        raise RuntimeError("No model was trained.")

    return best_model, best_metrics


def save_model_artifacts(output_dir: Path = ARTIFACTS_MODELS_DIR) -> tuple[Path, Path, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, metrics = train_and_log_models()

    dataset = load_training_dataset()
    metadata = {
        "feature_order": FEATURE_COLUMNS,
        "crops": sorted(dataset["Item"].dropna().unique().tolist()),
        "areas": sorted(dataset["Area"].dropna().unique().tolist()),
        "metrics": metrics,
    }

    model_path = output_dir / "model.joblib"
    metadata_path = output_dir / "metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path, metrics


if __name__ == "__main__":
    model_path, metadata_path, metrics = save_model_artifacts()
    print(f"model={model_path}")
    print(f"metadata={metadata_path}")
    print(json.dumps(metrics, indent=2))
