from typing import List

from pydantic import BaseModel, Field


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


class PredictionBatchInput(BaseModel):

    inputs: List[PredictionInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of 1 to 100 prediction inputs"
    )


class PredictionBatchOutput(BaseModel):

    predictions: List[PredictionOutput]