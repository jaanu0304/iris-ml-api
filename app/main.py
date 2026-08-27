from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import joblib
import numpy as np
import uuid
from app.models.schemas import PredictionInput, PredictionOutput

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = joblib.load("ml/saved_model/model.joblib")
    print("ML model loaded successfully!")

    yield


app = FastAPI(lifespan=lifespan)

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal value error occurred"
        }
    )


@app.get("/")
def root():
    return {"message": "ML API is alive"}



@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }





@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
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
        print(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    request_id = str(uuid.uuid4())

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "model_version": "1.0",
        "request_id": request_id
    }