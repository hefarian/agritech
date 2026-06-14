from contextlib import contextmanager
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from agritech.models import train


def _make_training_frame(rows: int = 40) -> pd.DataFrame:
    areas = ["Albania", "France"]
    items = ["Wheat", "Maize"]
    records = []
    for idx in range(rows):
        records.append(
            {
                "Area": areas[idx % len(areas)],
                "Item": items[idx % len(items)],
                "Year": 1990 + (idx % 20),
                "average_rain_fall_mm_per_year": 800.0 + idx,
                "pesticides_tonnes": 100.0 + idx,
                "avg_temp": 10.0 + (idx % 15),
                "hg_ha_yield": 10000.0 + (idx * 50.0),
            }
        )
    return pd.DataFrame.from_records(records)


def test_load_training_dataset_builds_if_missing(monkeypatch, tmp_path) -> None:
    # Si le CSV consolide n'existe pas, la fonction doit declencher la preparation.
    dataset = _make_training_frame(rows=5)
    monkeypatch.setattr(train, "ARTIFACTS_DATA_DIR", tmp_path)

    def fake_save_master_dataset():
        path = tmp_path / "consolidated_yield.csv"
        dataset.to_csv(path, index=False)
        return path, tmp_path / "merge_report.json"

    monkeypatch.setattr(train, "save_master_dataset", fake_save_master_dataset)
    loaded = train.load_training_dataset()
    assert len(loaded) == 5
    assert set(train.FEATURE_COLUMNS).issubset(set(loaded.columns))


def test_build_pipeline_and_evaluate_predictions() -> None:
    # Le pipeline doit assembler preprocessing + modele puis produire les metriques attendues.
    pipeline = train.build_pipeline(RandomForestRegressor(n_estimators=5, random_state=42))
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps.keys()) == ["preprocessor", "model"]

    metrics = train.evaluate_predictions(pd.Series([1.0, 2.0]), [1.0, 2.0])
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_log_model_run_emits_mlflow_calls(monkeypatch) -> None:
    # Le logger doit envoyer params et metriques cles a MLflow.
    calls = {"params": [], "metrics": []}

    monkeypatch.setattr(train.mlflow, "log_param", lambda key, value: calls["params"].append((key, value)))
    monkeypatch.setattr(train.mlflow, "log_params", lambda values: calls["params"].append(("bulk", values)))
    monkeypatch.setattr(train.mlflow, "log_metrics", lambda values: calls["metrics"].append(("bulk", values)))
    monkeypatch.setattr(train.mlflow, "log_metric", lambda key, value: calls["metrics"].append((key, value)))

    profitability_stub = {
        "profit_score": 0.96,
        "under_pred_rate": 0.3,
        "under_pred_loss_pct": 4.0,
        "mean_rel_error_pct": 5.0,
    }
    train._log_model_run(
        "random_forest",
        baseline_metrics={"rmse": 10.0, "mae": 5.0, "r2": 0.8, **profitability_stub},
        tuned_metrics={"rmse": 9.0, "mae": 4.0, "r2": 0.85, **profitability_stub},
        best_params={"n_estimators": 10},
    )

    assert ("model_name", "random_forest") in calls["params"]
    assert any(item[0] == "baseline_rmse" for item in calls["metrics"])
    assert any(item[0] == "tuned_rmse" for item in calls["metrics"])


def test_train_and_log_models_returns_pipeline_and_summary(monkeypatch, tmp_path) -> None:
    # Entrainement de bout en bout avec un jeu synthetique et un logging MLflow simule.
    monkeypatch.setattr(train, "load_training_dataset", lambda _: _make_training_frame(rows=60))
    monkeypatch.setattr(train, "MLRUNS_DIR", tmp_path / "mlruns")
    monkeypatch.setattr(
        train,
        "NOTEBOOK3_RF_STAGE2_PARAMS",
        {
            "n_estimators": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "max_depth": None,
            "bootstrap": False,
        },
    )

    monkeypatch.setattr(train.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(train.mlflow, "set_experiment", lambda name: None)

    @contextmanager
    def fake_start_run(run_name=None):
        yield {"run_name": run_name}

    monkeypatch.setattr(train.mlflow, "start_run", fake_start_run)

    log_capture = {}
    monkeypatch.setattr(
        train,
        "_log_model_run",
        lambda model_name, baseline_metrics, tuned_metrics, best_params: log_capture.update(
            {
                "model_name": model_name,
                "baseline_rmse": baseline_metrics["rmse"],
                "tuned_rmse": tuned_metrics["rmse"],
                "params": best_params,
            }
        ),
    )

    model, summary = train.train_and_log_models()

    assert isinstance(model, Pipeline)
    assert summary["model_name"] == "random_forest"
    assert "rmse" in summary
    assert "baseline_rmse" in summary
    assert log_capture["model_name"] == "random_forest"


def test_save_model_artifacts_writes_model_and_metadata(monkeypatch, tmp_path) -> None:
    # Les artefacts de sortie doivent inclure le modele pickled et les metadonnees JSON.
    metrics = {"rmse": 1.0, "mae": 1.0, "r2": 0.9}
    monkeypatch.setattr(train, "train_and_log_models", lambda: ({"kind": "fake-model"}, metrics))
    monkeypatch.setattr(
        train,
        "load_training_dataset",
        lambda: pd.DataFrame(
            [
                {"Area": "France", "Item": "Wheat"},
                {"Area": "Albania", "Item": "Maize"},
            ]
        ),
    )

    model_path, metadata_path, saved_metrics = train.save_model_artifacts(output_dir=tmp_path)

    assert model_path.exists()
    assert metadata_path.exists()
    assert saved_metrics == metrics

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["feature_order"] == train.FEATURE_COLUMNS
    assert metadata["areas"] == ["Albania", "France"]
    assert metadata["crops"] == ["Maize", "Wheat"]
