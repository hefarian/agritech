from fastapi.testclient import TestClient

from api.main import create_app


class DummyPredictor:
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
