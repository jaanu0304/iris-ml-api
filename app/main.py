from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import joblib

import uuid
import time

from app.logging_config import setup_logging
from app.routers.v1 import router as v1_router
from app.config import settings

model = None
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = joblib.load(settings.MODEL_PATH)
    logger.info("ML model loaded successfully")

    yield


app = FastAPI(
    title=settings.API_TITLE,
    lifespan=lifespan
)

app.state.logger = logger

app.include_router(v1_router)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    logger.info(
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status_code={response.status_code} "
        f"duration={duration:.4f}s"
    )

    response.headers["X-Request-ID"] = request_id

    return response

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal value error occurred"
        }
    )






