from fastapi import APIRouter, HTTPException, Request
import numpy as np

from app.models.schemas import PredictionInput, V2PredictionOutput
from app.config import settings


router = APIRouter(prefix="/api/v2")


@router.post("/predict", response_model=V2PredictionOutput)
def predict_v2(data: PredictionInput, request: Request):
    from app.main import model

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    logger = request.app.state.logger
    request_id = request.state.request_id

    try:
        prediction = model.predict(features)[0]
        probabilities_array = model.predict_proba(features)[0]
        class_names = model.classes_

        probabilities = {
            str(class_name): float(probability)
            for class_name, probability
            in zip(class_names, probabilities_array)
        }

    except Exception as e:
        logger.error(
            f"request_id={request_id} "
            f"v2_prediction_failed=true "
            f"error={e}"
        )

        raise HTTPException(
            status_code=500,
            detail="V2 prediction failed"
        )

    logger.info(
        f"request_id={request_id} "
        f"v2_prediction_success=true "
        f"prediction={prediction}"
    )

    return {
        "prediction": str(prediction),
        "probabilities": probabilities,
        "model_version": settings.MODEL_VERSION,
        "request_id": request_id
    }