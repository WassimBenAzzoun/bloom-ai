from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import json
import re
import unicodedata
from functools import lru_cache
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIG
# ==============================
ARTIFACT_DIR = "models"
CONFIDENCE_THRESHOLD = 0.15  # uncertainty threshold

# ==============================
# LOAD ARTIFACTS
# ==============================
model = joblib.load(f"{ARTIFACT_DIR}/model_xgb.joblib")
scaler = joblib.load(f"{ARTIFACT_DIR}/num_scaler.joblib")
label_encoder = joblib.load(f"{ARTIFACT_DIR}/label_encoder.joblib")
saison_encoder = joblib.load(f"{ARTIFACT_DIR}/saison_encoder.joblib")
vent_encoder = joblib.load(f"{ARTIFACT_DIR}/vent_encoder.joblib")

with open(f"{ARTIFACT_DIR}/embedder_config.json", "r", encoding="utf-8") as f:
    EMBEDDER_NAME = json.load(f)["embedder_name"]

embedder = SentenceTransformer(EMBEDDER_NAME, device="cpu")

# ==============================
# FASTAPI APP
# ==============================
app = FastAPI(
    title="BLOOM-VENOM Predictor",
    version="1.0.0",
    description="Apiculture diagnostic decision support (CPU-safe)"
)

# ==============================
# TEXT NORMALIZATION
# ==============================
META_PHRASES = [
    "observation additionnelle",
    "observation supplementaire",
    "remarque",
    "note",
    "commentaire",
]

def normalize_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    for p in META_PHRASES:
        t = t.replace(p, " ")
    t = re.sub(r"[.:;,\n\r\t]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def normalize_for_encoder(value: str, encoder):
    value_norm = normalize_text(str(value))
    for cls in encoder.classes_:
        if normalize_text(cls) == value_norm:
            return cls
    raise HTTPException(
        status_code=400,
        detail=f"Unknown category '{value}'. Allowed: {list(encoder.classes_)}"
    )

# ==============================
# EMBEDDING CACHE (CRITICAL)
# ==============================
@lru_cache(maxsize=512)
def embed_cached(text: str) -> np.ndarray:
    text = normalize_text(text)
    vec = embedder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]
    return vec.astype(np.float32)

# ==============================
# REQUEST / RESPONSE MODELS
# ==============================
class PredictRequest(BaseModel):
    text: str
    temperature: float
    humidite: float
    poids_ruche: float
    saison: str
    vent_nuages: str
    top_k: int = 3

class Prediction(BaseModel):
    scenariokey: str
    confidence: float

class PredictResponse(BaseModel):
    status: str
    predictions: List[Prediction]

# ==============================
# ENDPOINTS
# ==============================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    # ---- Text embedding
    emb = np.array([embed_cached(req.text)], dtype=np.float32)

    # ---- Encode categorical features
    saison_fixed = normalize_for_encoder(req.saison, saison_encoder)
    vent_fixed = normalize_for_encoder(req.vent_nuages, vent_encoder)

    saison_code = int(saison_encoder.transform([saison_fixed])[0])
    vent_code = int(vent_encoder.transform([vent_fixed])[0])

    # ---- Numerical vector
    num = np.array([[
        req.temperature,
        req.humidite,
        req.poids_ruche,
        saison_code,
        vent_code
    ]], dtype=np.float32)

    num = scaler.transform(num).astype(np.float32)
    num = np.nan_to_num(num, nan=0.0, posinf=5.0, neginf=-5.0)

    # ---- Final input
    X = np.hstack([emb, num]).astype(np.float32)

    # ---- Predict
    proba = model.predict_proba(X)[0]
    top_idx = np.argsort(proba)[-req.top_k:][::-1]

    predictions = [
        {
            "scenariokey": label_encoder.inverse_transform([i])[0],
            "confidence": float(proba[i])
        }
        for i in top_idx
    ]

    # ---- Uncertainty handling
    if predictions[0]["confidence"] < CONFIDENCE_THRESHOLD:
        return {
            "status": "UNCERTAIN",
            "predictions": predictions
        }

    return {
        "status": "OK",
        "predictions": predictions
    }
