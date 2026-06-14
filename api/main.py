import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agritech.models.predictor import CropYieldPredictor


class PlotContext(BaseModel):
    # Champs communs utilises a la fois pour la prediction et la recommandation.
    Area: str = Field(..., examples=["Albania"])
    Year: int = Field(..., ge=1990, le=2100)
    average_rain_fall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float


class PredictionRequest(PlotContext):
    # La prediction a besoin d'un champ de plus : la culture a evaluer.
    Item: str


class RecommendationRequest(PlotContext):
    pass


def create_app(predictor: CropYieldPredictor | None = None) -> FastAPI:
    # On stocke le predictor dans un petit etat local pour que les tests puissent injecter un faux modele.
    state = {"predictor": predictor}
    project_name = os.getenv("COMPOSE_PROJECT_NAME", "local")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Startup leger : le chargement du modele est fait a la demande.
        yield

    app = FastAPI(
        title=f"Agritech Answers API - {project_name}",
        description="API de prediction de rendement et de recommandation de culture.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def get_predictor() -> CropYieldPredictor:
        # Chargement lazy pour eviter de bloquer le demarrage si les artefacts sont absents/incompatibles.
        active_predictor = state["predictor"]
        if active_predictor is None:
            try:
                active_predictor = CropYieldPredictor.from_artifacts()
                state["predictor"] = active_predictor
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Model artifacts unavailable or incompatible. Run training first.",
                ) from exc
        if active_predictor is None:
            raise HTTPException(status_code=503, detail="Model artifacts not found. Run training first.")
        return active_predictor

    @app.get("/health")
    def health() -> dict:
        # Endpoint simple pour les humains, les tests et les checks de deploiement.
        return {"status": "ok", "model_loaded": state["predictor"] is not None}

    @app.get("/metadata")
    def metadata() -> dict:
        # Le front utilise ceci pour remplir dynamiquement les listes deroulantes.
        active_predictor = get_predictor()
        return active_predictor.metadata

    @app.post("/predict")
    def predict(payload: PredictionRequest) -> dict:
        # On calcule une prediction pour une seule culture.
        active_predictor = get_predictor()
        prediction = active_predictor.predict_one(payload.model_dump())
        return {"predicted_yield": prediction, "unit": "hg/ha"}

    @app.post("/recommend")
    def recommend(payload: RecommendationRequest) -> dict:
        # On classe toutes les cultures connues pour un meme contexte de parcelle.
        active_predictor = get_predictor()
        ranking = active_predictor.recommend(payload.model_dump())
        return {"recommendations": ranking, "unit": "hg/ha"}

    return app


# Objet d'application pret a etre lance directement par Uvicorn.
app = create_app()
