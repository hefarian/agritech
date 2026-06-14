import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from agritech.config import ARTIFACTS_MODELS_DIR


@dataclass
class CropYieldPredictor:
    # `pipeline` contient a la fois le preprocessing et le modele entraine.
    pipeline: object
    metadata: dict

    @property
    def crops(self) -> list[str]:
        # L'endpoint de recommandation reutilise cette liste pour tester chaque culture connue.
        return list(self.metadata["crops"])

    def predict_records(self, frame: pd.DataFrame) -> list[float]:
        # On convertit les valeurs NumPy en floats Python simples pour faciliter le JSON.
        predictions = self.pipeline.predict(frame)
        return [float(value) for value in predictions]

    def predict_one(self, features: dict) -> float:
        # On reconstruit un dataframe d'une ligne avec l'ordre exact attendu par la pipeline.
        frame = pd.DataFrame([features], columns=self.metadata["feature_order"])
        return self.predict_records(frame)[0]

    def recommend(self, context: dict) -> list[dict]:
        # Pour la recommandation, le contexte reste fixe et seule la culture change.
        rows = []
        for crop in self.crops:
            row = {**context, "Item": crop}
            rows.append(row)
        frame = pd.DataFrame(rows, columns=self.metadata["feature_order"])
        predictions = self.predict_records(frame)
        ranking = [
            {"Item": row["Item"], "predicted_yield": prediction}
            for row, prediction in zip(rows, predictions, strict=True)
        ]
        ranking.sort(key=lambda item: item["predicted_yield"], reverse=True)
        return ranking

    @classmethod
    def from_artifacts(
        cls,
        model_path: Path = ARTIFACTS_MODELS_DIR / "model.joblib",
        metadata_path: Path = ARTIFACTS_MODELS_DIR / "metadata.json",
    ) -> "CropYieldPredictor":
        # Charger les deux fichiers separe le modele lui-meme de ses metadonnees metier.
        pipeline = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(pipeline=pipeline, metadata=metadata)
