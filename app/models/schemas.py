from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Must be positive")
    sepal_width: float = Field(..., gt=0, description="Must be positive")
    petal_length: float = Field(..., gt=0, description="Must be positive")
    petal_width: float = Field(..., gt=0, description="Must be positive")