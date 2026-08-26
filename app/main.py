from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib
import numpy as np
import uuid
from app.models.schemas import PredictionInput


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = joblib.load("ml/saved_model/model.joblib")
    print("ML model loaded successfully!")

    yield


app = FastAPI(lifespan=lifespan)




@app.get("/")
def root():
    return {"message": "ML API is alive"}



@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }





@app.post("/predict")
def predict(data: PredictionInput):
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(features)[0]

    confidence = model.predict_proba(features).max()

    request_id = str(uuid.uuid4())

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "request_id": request_id
    }