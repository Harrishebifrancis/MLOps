# src/main.py
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ALIASES = {
    # any of these in the model bundle will be read from the underscored key in JSON
    "od280/od315 of diluted wines": "od280_od315_of_diluted_wines",
    "od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines",
    "od280_od315_of_diluted_wines": "od280_od315_of_diluted_wines",  # idempotent
}

def resolve_key(name: str) -> str:
    return ALIASES.get(name, name)

app = FastAPI(title="Wine Classifier API")

# let Swagger UI (and anything on localhost) call your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Relative import is fine when running as a package (uvicorn src.main:app)
try:
    from .data import (
        WineData, WineResponse, WineProbaResponse,
        MinimalInput, MinimalResponse
    )
except ImportError:
    # Fallback for running as a script inside src/ (uvicorn main:app)
    from data import (
        WineData, WineResponse, WineProbaResponse,
        MinimalInput, MinimalResponse
    )

app = FastAPI(title="Wine Classifier API")

# ---- Load model once ----
# Robust path: /project_root/model/wine_model.pkl
MODEL_PATH = (Path(__file__).resolve().parents[1] / "model" / "wine_model.pkl")
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run train.py first.")

bundle: Dict = joblib.load(MODEL_PATH)
FULL_MODEL = bundle["full_model"]
MIN_MODEL = bundle["minimal_model"]
FEATURE_NAMES: List[str] = bundle["feature_names"]
SELECTED_NAMES: List[str] = bundle["selected_names"]
CLASS_NAMES: List[str] = bundle["class_names"]

# ---- Compatibility helpers (Pydantic v1 & v2) ----
def model_to_dict(obj) -> Dict[str, float]:
    # v2: model_dump, v1: dict
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj.dict()

# ---- Vectorizers ----
def vector_from_wine(w: WineData) -> np.ndarray:
    data_dict = model_to_dict(w)
    try:
        vals = [float(data_dict[resolve_key(name)]) for name in FEATURE_NAMES]
    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"Missing feature: {ke}")
    return np.array(vals, dtype=float).reshape(1, -1)

def vector_from_min(features: Dict[str, float]) -> np.ndarray:
    try:
        vals = [float(features[resolve_key(name)]) for name in SELECTED_NAMES]
    except KeyError as ke:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required feature for minimal model: {ke}. "
                   f"Expected keys: {SELECTED_NAMES}"
        )
    return np.array(vals, dtype=float).reshape(1, -1)


@app.get("/")
def health_ping():
    return {
        "message": "Wine Classifier is up",
        "full_features": FEATURE_NAMES,
        "minimal_features": SELECTED_NAMES,
        "classes": CLASS_NAMES,
    }

@app.post("/predict", response_model=WineResponse)
def predict_full(payload: WineData):
    X = vector_from_wine(payload)
    label = int(FULL_MODEL.predict(X)[0])
    return {"response": label}



@app.post("/predict_min", response_model=MinimalResponse)
def predict_minimal(payload: MinimalInput):
    X = vector_from_min(payload.features)
    proba = MIN_MODEL.predict_proba(X)[0].tolist()
    label = int(np.argmax(proba))
    return {
        "predicted_label": label,
        "probabilities": proba,
        "used_features": SELECTED_NAMES,
        "class_names": CLASS_NAMES,
    }

# Optional: quick self-check on startup
@app.on_event("startup")
def _startup_check():
    # build a tiny zero vector with correct shape just to hit predict once
    try:
        _ = FULL_MODEL.predict(np.zeros((1, len(FEATURE_NAMES))))
        _ = MIN_MODEL.predict(np.zeros((1, len(SELECTED_NAMES))))
    except Exception as e:
        raise RuntimeError(f"Model compatibility issue: {e}")
