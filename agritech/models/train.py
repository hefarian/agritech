import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
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

NOTEBOOK3_RF_STAGE2_PARAMS = {
    "n_estimators": 800,
    "min_samples_split": 3,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "max_depth": None,
    "bootstrap": False,
}

NOTEBOOK3_RF_STAGE2_TEST_METRICS = {
    "rmse": 7560.176434709961,
    "mae": 2978.9974526996225,
    "r2": 0.9939772766482966,
}


def load_training_dataset(data_path: Path | None = None) -> pd.DataFrame:
    # Si le dataset nettoye n'existe pas encore, on le construit automatiquement.
    if data_path is None:
        data_path = ARTIFACTS_DATA_DIR / "consolidated_yield.csv"
    if not data_path.exists():
        save_master_dataset()
    return pd.read_csv(data_path)


def build_pipeline(model: object) -> Pipeline:
    # Les colonnes categorielles et numeriques ont besoin de traitements differents.
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
    # On derive la RMSE de la MSE car elle se lit plus facilement dans l'unite cible.
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def _log_model_run(model_name: str, baseline_metrics: dict, tuned_metrics: dict, best_params: dict) -> None:
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("search_type", "notebook3_fixed_params")
    mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
    mlflow.log_params(best_params)
    mlflow.log_metrics({f"notebook3_expected_{k}": v for k, v in NOTEBOOK3_RF_STAGE2_TEST_METRICS.items()})

    mlflow.log_metric("baseline_rmse", baseline_metrics["rmse"])
    mlflow.log_metric("baseline_mae", baseline_metrics["mae"])
    mlflow.log_metric("baseline_r2", baseline_metrics["r2"])
    mlflow.log_metric("tuned_rmse", tuned_metrics["rmse"])
    mlflow.log_metric("tuned_mae", tuned_metrics["mae"])
    mlflow.log_metric("tuned_r2", tuned_metrics["r2"])


def train_and_log_models(data_path: Path | None = None) -> tuple[Pipeline, dict]:
    # On separe les variables explicatives de la cible avant l'entrainement.
    dataset = load_training_dataset(data_path)
    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]

    # La seed est fixee pour rendre les resultats reproductibles.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("agritech-yield")

    # On conserve un baseline et on applique ensuite les hyperparametres optimises du notebook 3.
    baseline_pipeline = build_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1))
    baseline_pipeline.fit(x_train, y_train)
    baseline_predictions = baseline_pipeline.predict(x_test)
    baseline_metrics = evaluate_predictions(y_test, baseline_predictions)

    final_pipeline = build_pipeline(
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            n_estimators=NOTEBOOK3_RF_STAGE2_PARAMS["n_estimators"],
            min_samples_split=NOTEBOOK3_RF_STAGE2_PARAMS["min_samples_split"],
            min_samples_leaf=NOTEBOOK3_RF_STAGE2_PARAMS["min_samples_leaf"],
            max_features=NOTEBOOK3_RF_STAGE2_PARAMS["max_features"],
            max_depth=NOTEBOOK3_RF_STAGE2_PARAMS["max_depth"],
            bootstrap=NOTEBOOK3_RF_STAGE2_PARAMS["bootstrap"],
        )
    )
    final_pipeline.fit(x_train, y_train)
    final_predictions = final_pipeline.predict(x_test)
    final_metrics = evaluate_predictions(y_test, final_predictions)

    with mlflow.start_run(run_name="random_forest_notebook3_stage2"):
        _log_model_run("random_forest", baseline_metrics, final_metrics, NOTEBOOK3_RF_STAGE2_PARAMS)

    training_summary = final_metrics | {
        "model_name": "random_forest",
        "best_params": NOTEBOOK3_RF_STAGE2_PARAMS,
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_mae": baseline_metrics["mae"],
        "baseline_r2": baseline_metrics["r2"],
        "notebook3_expected_metrics": NOTEBOOK3_RF_STAGE2_TEST_METRICS,
    }
    return final_pipeline, training_summary


def save_model_artifacts(output_dir: Path = ARTIFACTS_MODELS_DIR) -> tuple[Path, Path, dict]:
    # On sauvegarde la pipeline entrainee et les metadonnees utiles a l'API et au front.
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
