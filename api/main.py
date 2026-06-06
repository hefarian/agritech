from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agritech.models.predictor import CropYieldPredictor


class PlotContext(BaseModel):
    Area: str = Field(..., examples=["Albania"])
    Year: int = Field(..., ge=1990, le=2100)
    average_rain_fall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float


class PredictionRequest(PlotContext):
    Item: str


class RecommendationRequest(PlotContext):
    pass


def create_app(predictor: CropYieldPredictor | None = None) -> FastAPI:
    state = {"predictor": predictor}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["predictor"] is None:
            try:
                state["predictor"] = CropYieldPredictor.from_artifacts()
            except FileNotFoundError:
                state["predictor"] = None
        yield

    app = FastAPI(
        title="Agritech Answers API",
        description="API de prediction de rendement et de recommandation de culture.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def get_predictor() -> CropYieldPredictor:
        active_predictor = state["predictor"]
        if active_predictor is None:
            raise HTTPException(status_code=503, detail="Model artifacts not found. Run training first.")
        return active_predictor

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": state["predictor"] is not None}

    @app.get("/metadata")
    def metadata() -> dict:
        active_predictor = get_predictor()
        return active_predictor.metadata

    @app.post("/predict")
    def predict(payload: PredictionRequest) -> dict:
        active_predictor = get_predictor()
        prediction = active_predictor.predict_one(payload.model_dump())
        return {"predicted_yield": prediction, "unit": "hg/ha"}

    @app.post("/recommend")
    def recommend(payload: RecommendationRequest) -> dict:
        active_predictor = get_predictor()
        ranking = active_predictor.recommend(payload.model_dump())
        return {"recommendations": ranking, "unit": "hg/ha"}

    return app


app = create_app()
