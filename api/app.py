"""
api/app.py
----------
FastAPI REST API for Titanic survival prediction.

Endpoints
---------
GET  /         → Health check
GET  /info     → Model metadata + evaluation metrics
POST /predict  → Survival prediction
GET  /docs     → Auto-generated Swagger UI  (FastAPI built-in)
GET  /redoc    → Auto-generated ReDoc UI    (FastAPI built-in)

Run locally
-----------
    uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.schemas import (
    PassengerInput, PredictionResponse, HealthResponse, InfoResponse,
)
from src.predict import predict, load_model, load_metadata
from src.preprocessor import (
    NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, RAW_INPUT_FEATURES,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan: warm model cache at startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
        logger.info("Model loaded and cached successfully")
    except FileNotFoundError:
        logger.warning("Model file not found — /predict will return 503 until trained")
    yield


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Titanic Survival Prediction API",
    description=(
        "Production-ready ML API that predicts whether a Titanic passenger survived.\n\n"
        "**Model**: RandomForestClassifier with engineered features (FamilySize, IsAlone, Title)\n\n"
        "**Tuning**: RandomizedSearchCV (40 iterations, 5-fold stratified CV)\n\n"
        "**Threshold**: Optimised on a validation set to maximise F1-score"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Middleware: request logging ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("%s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("→ %d", response.status_code)
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health_check():
    """
    Returns service health status.
    - **status**: `ok` if the model is loaded, `degraded` otherwise
    - **model_ready**: whether inference is available
    """
    try:
        load_model()
        model_ready = True
    except FileNotFoundError:
        model_ready = False

    return HealthResponse(
        status      = "ok" if model_ready else "degraded",
        service     = "titanic-ml-api",
        version     = "2.0.0",
        model_ready = model_ready,
    )


@app.get(
    "/info",
    response_model=InfoResponse,
    summary="Model info & metrics",
    tags=["System"],
)
def model_info():
    """
    Returns model metadata including:
    - Feature schema (which columns the model expects)
    - Valid categorical values
    - Training evaluation metrics (accuracy, ROC-AUC, threshold, …)
    """
    meta = load_metadata()
    return InfoResponse(
        model   = "RandomForestClassifier (sklearn Pipeline)",
        version = "2.0.0",
        features = {
            "numeric":     NUMERIC_FEATURES,
            "binary":      BINARY_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "raw_input":   RAW_INPUT_FEATURES,
        },
        valid_values = {
            "Pclass":   [1, 2, 3],
            "Sex":      ["male", "female"],
            "Embarked": ["C", "Q", "S"],
        },
        training_metrics = meta,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict passenger survival",
    tags=["Prediction"],
    responses={
        200: {"description": "Prediction returned successfully"},
        422: {"description": "Invalid input — see detail for field-level errors"},
        503: {"description": "Model not trained yet"},
    },
)
def predict_endpoint(passenger: PassengerInput):
    """
    Predict whether a Titanic passenger would have survived.

    **Required fields**: `Pclass`, `Sex`, `Fare`

    **Optional fields**: `Age`, `Embarked`, `SibSp`, `Parch`
    (missing values are handled by the pipeline's imputation step)

    Returns:
    - `prediction`: **0** = Did Not Survive, **1** = Survived
    - `probability`: survival probability (0.0–1.0)
    - `label`: human-readable string
    - `threshold`: decision threshold used (optimised for F1)
    """
    try:
        result = predict(passenger.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal prediction error")

    return PredictionResponse(
        **result,
        input=passenger.model_dump(exclude_none=False),
    )


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": f"Endpoint '{request.url.path}' not found. "
                           "Available: GET /, GET /info, POST /predict, GET /docs"},
    )
