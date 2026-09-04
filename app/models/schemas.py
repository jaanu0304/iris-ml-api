from typing import List

from pydantic import BaseModel, Field

from app.config import settings


class PredictionInput(BaseModel):

    sepal_length: float = Field(..., gt=0, description="Must be positive")

    sepal_width: float = Field(..., gt=0, description="Must be positive")

    petal_length: float = Field(..., gt=0, description="Must be positive")

    petal_width: float = Field(..., gt=0, description="Must be positive")


class PredictionOutput(BaseModel):

    prediction: str

    confidence: float

    model_version: str

    request_id: str


class V2PredictionOutput(BaseModel):

    prediction: str

    probabilities: dict[str, float]

    model_version: str

    request_id: str


class PredictionBatchInput(BaseModel):

    inputs: List[PredictionInput] = Field(
        ...,
        min_length=1,
        max_length=settings.MAX_BATCH_SIZE,
        description="List of prediction inputs within the configured batch size limit"
    )


class PredictionBatchOutput(BaseModel):

    predictions: List[PredictionOutput]