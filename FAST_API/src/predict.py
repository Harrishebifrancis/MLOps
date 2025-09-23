from pathlib import Path
import joblib
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "wine_model.pkl"
_payload:Optional[ Dict[str, Any]] = None

def load_payload():
    global _payload
    if _payload is None:
        _payload = joblib.load(MODEL_PATH)
    return _payload

def feature_names() -> List[str]:
    return load_payload()["feature_names"]

def selected_names() -> List[str]:
    return load_payload()["selected_names"]

def class_names() -> List[str]:
    return load_payload()["class_names"]

def predict_full_row(values_13: List[float]) -> Tuple[int, List[float]]:
    pipe = load_payload()["full_model"]
    X = np.asarray([values_13], dtype=float)
    proba = pipe.predict_proba(X)[0].tolist()
    label = int(np.argmax(proba))
    return label, [float(p) for p in proba]

def predict_minimal_map(feat_map: Dict[str, float]) -> Tuple[int, List[float], List[str]]:
    names = selected_names()
    missing = [n for n in names if n not in feat_map]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    X = np.asarray([[feat_map[n] for n in names]], dtype=float)
    pipe = load_payload()["minimal_model"]
    proba = pipe.predict_proba(X)[0].tolist()
    label = int(np.argmax(proba))
    return label, [float(p) for p in proba], names
