from fastapi.testclient import TestClient

import api.main as api_main
from api.main import create_app


class DummyPredictor:
    # Ce faux predictor permet de tester l'API sans dependre du vrai modele de ML.
    metadata = {
        "feature_order": [
            "Area",
            "Item",
            "Year",
            "average_rain_fall_mm_per_year",
            "pesticides_tonnes",
            "avg_temp",
        ],
        "crops": ["Maize", "Wheat"],
        "areas": ["Albania"],
    }

    def predict_one(self, features: dict) -> float:
        return 12345.0 if features["Item"] == "Wheat" else 11000.0

    def recommend(self, context: dict) -> list[dict]:
        return [
            {"Item": "Wheat", "predicted_yield": 12345.0},
            {"Item": "Maize", "predicted_yield": 11000.0},
        ]


def test_predict_endpoint_returns_prediction() -> None:
    # L'endpoint doit transmettre les donnees au predictor et renvoyer le resultat en JSON.
    client = TestClient(create_app(DummyPredictor()))
    response = client.post(
        "/predict",
        json={
            "Area": "Albania",
            "Item": "Wheat",
            "Year": 2013,
            "average_rain_fall_mm_per_year": 1485.0,
            "pesticides_tonnes": 121.0,
            "avg_temp": 16.37,
        },
    )
    assert response.status_code == 200
    assert response.json()["predicted_yield"] == 12345.0


def test_recommend_endpoint_returns_sorted_ranking() -> None:
    # La recommandation doit renvoyer d'abord la culture au rendement le plus eleve.
    client = TestClient(create_app(DummyPredictor()))
    response = client.post(
        "/recommend",
        json={
            "Area": "Albania",
            "Year": 2013,
            "average_rain_fall_mm_per_year": 1485.0,
            "pesticides_tonnes": 121.0,
            "avg_temp": 16.37,
        },
    )
    assert response.status_code == 200
    assert response.json()["recommendations"][0]["Item"] == "Wheat"


def test_metadata_endpoint_returns_model_metadata() -> None:
    # Le endpoint metadata expose les listes utiles au front.
    client = TestClient(create_app(DummyPredictor()))
    response = client.get("/metadata")
    assert response.status_code == 200
    payload = response.json()
    assert payload["crops"] == ["Maize", "Wheat"]
    assert payload["areas"] == ["Albania"]


def test_predict_validation_rejects_year_before_1990() -> None:
    # Le schema Pydantic doit bloquer les annees hors bornes metier.
    client = TestClient(create_app(DummyPredictor()))
    response = client.post(
        "/predict",
        json={
            "Area": "Albania",
            "Item": "Wheat",
            "Year": 1980,
            "average_rain_fall_mm_per_year": 1485.0,
            "pesticides_tonnes": 121.0,
            "avg_temp": 16.37,
        },
    )
    assert response.status_code == 422


def test_health_and_metadata_when_model_is_unavailable(monkeypatch) -> None:
    # Si les artefacts manquent au demarrage, la sante est OK mais sans modele charge.
    monkeypatch.setattr(api_main.CropYieldPredictor, "from_artifacts", lambda: (_ for _ in ()).throw(FileNotFoundError()))

    with TestClient(create_app()) as client:
        health_response = client.get("/health")
        metadata_response = client.get("/metadata")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok", "model_loaded": False}
    assert metadata_response.status_code == 503
