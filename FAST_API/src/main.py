# src/main.py
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Schemas
try:
    from .data import WineData, WineProbaResponse
except ImportError:
    from data import WineData, WineProbaResponse

app = FastAPI(title="Wine Classifier API")

# CORS for Swagger/browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model bundle ----
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "wine_model.pkl"
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run train.py first.")
bundle: Dict = joblib.load(MODEL_PATH)

FULL_MODEL = bundle["full_model"]
FEATURE_NAMES: List[str] = bundle["feature_names"]
CLASS_NAMES: List[str] = bundle["class_names"]

# Accept either slash or underscore key for the tricky feature
ALIASES = {
    "od280/od315 of diluted wines": "od280_od315_of_diluted_wines",
    "od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines",
    "od280_od315_of_diluted_wines": "od280_od315_of_diluted_wines",
}
def resolve_key(name: str) -> str:
    return ALIASES.get(name, name)

def model_to_dict(obj) -> Dict[str, float]:
    # pydantic v2 (model_dump) or v1 (dict)
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()

def vector_from_wine(w: WineData) -> np.ndarray:
    data = model_to_dict(w)
    try:
        vals = [float(data[resolve_key(name)]) for name in FEATURE_NAMES]
    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"Missing feature: {ke}")
    return np.array(vals, dtype=float).reshape(1, -1)

@app.get("/")
def root():
    return {"message": "Wine Classifier is up",
            "full_features": FEATURE_NAMES,
            "classes": CLASS_NAMES}

@app.post("/predict", response_model=WineProbaResponse)
def predict(payload: WineData):
    X = vector_from_wine(payload)
    proba = FULL_MODEL.predict_proba(X)[0].tolist()
    label = int(np.argmax(proba))
    return {"probabilities": proba, "class_names": CLASS_NAMES, "predicted_label": label}
