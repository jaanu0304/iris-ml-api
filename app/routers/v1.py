from fastapi import APIRouter, HTTPException, Request
import numpy as np

from app.models.schemas import PredictionInput, PredictionOutput


router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health(request: Request):
    from app.main import model

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@router.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput, request: Request):
    from app.main import model

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    try:
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features).max()

    except Exception as e:
        logger = request.app.state.logger

        logger.error(
            f"request_id={request.state.request_id} "
            f"prediction_failed=true "
            f"error={e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    request_id = request.state.request_id
    logger = request.app.state.logger

    logger.info(
        f"request_id={request_id} "
        f"prediction_success=true "
        f"prediction={prediction}"
    )

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "model_version": "1.0",
        "request_id": request_id
    }


# V2 plan:
# If /api/v2/predict is introduced, it should use a separate router
# and separate Pydantic response schema.
# Existing v1 clients will continue using the unchanged v1 contract.
# New fields can be added to the v2 response without breaking v1 clients.