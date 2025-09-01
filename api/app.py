from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any
import os, json
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
META_PATH = os.getenv("META_PATH", "model_meta.json")
DATA_PATH = os.getenv("DATA_PATH")  # optional: for /rank to read dataset

app = FastAPI(title="Hotel Cancellation Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + feature names
try:
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        META = json.load(f)
    FEATURE_NAMES = META["feature_names"]
except Exception as e:
    raise RuntimeError(f"Model or metadata not found: {e}")

class Record(BaseModel):
    # Accept dynamic features: we validate at runtime
    __root__: Dict[str, float]

class BatchRequest(BaseModel):
    records: List[Dict[str, float]]

class RankResponse(BaseModel):
    rows: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok", "model_features": len(FEATURE_NAMES)}

def _score_df(df: pd.DataFrame) -> pd.Series:
    # Ensure all features present; fill missing with 0
    for c in FEATURE_NAMES:
        if c not in df.columns:
            df[c] = 0.0
    df = df[FEATURE_NAMES]
    proba = model.predict_proba(df)[:, 1]
    return pd.Series(proba, index=df.index)

@app.post("/predict")
def predict(payload: Record):
    rec = payload.__root__
    df = pd.DataFrame([rec])
    proba = _score_df(df).iloc[0]
    return {"probability": float(proba)}

@app.post("/predict-batch")
def predict_batch(payload: BatchRequest):
    df = pd.DataFrame(payload.records)
    proba = _score_df(df)
    return {"probabilities": [float(x) for x in proba.values]}

@app.get("/rank", response_model=RankResponse)
def rank(
    top: int = Query(20, ge=1, le=200),
    arrival_window_days: int = Query(14, ge=1, le=365),
    alert_threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    if not DATA_PATH or not os.path.exists(DATA_PATH):
        raise HTTPException(500, detail="DATA_PATH not configured or file not found")
    df = pd.read_csv(DATA_PATH)
    # best effort: drop target if present
    if "booking_status" in df.columns:
        df = df.drop(columns=["booking_status"])

    # If there's a normalized 'arrival_date' column (0..1), map it to a date for UI purposes.
    today = datetime.utcnow().date()
    if "arrival_date" in df.columns:
        # interpret as 0..1 over 365 days horizon
        df["_arrival_dt"] = [today + timedelta(days=int(float(v)*365)) for v in df["arrival_date"]]
    else:
        # fallback: same date
        df["_arrival_dt"] = [today for _ in range(len(df))]

    # Compute scores
    scores = _score_df(df)
    df["_risk"] = scores

    # approximate window filter: keep arrivals within the next arrival_window_days
    df = df[df["_arrival_dt"] <= today + timedelta(days=arrival_window_days)]

    # choose columns to show if exist
    cols = []
    for c in ["Booking_ID", "arrival_dt", "no_of_adults", "avg_price_per_room", "no_of_special_requests"]:
        if c in df.columns:
            cols.append(c)
    # our synthetic columns if missing
    if "Booking_ID" not in df.columns:
        df["Booking_ID"] = df.index.astype(int) + 100000  # synthetic
        cols = ["Booking_ID"] + cols

    if "arrival_dt" not in df.columns:
        df["arrival_dt"] = df["_arrival_dt"].astype(str)
        if "arrival_dt" not in cols:
            cols.insert(1, "arrival_dt")

    cols = list(dict.fromkeys(cols))  # dedupe, keep order
    cols += ["_risk"]

    # sort by risk desc and filter by threshold
    out = df[df["_risk"] >= alert_threshold].sort_values("_risk", ascending=False).head(top)

    # format risk as float
    rows = []
    for _, r in out.iterrows():
        row = {}
        for k in cols:
            v = r[k]
            if isinstance(v, (float, int)):
                row[k] = float(v)
            else:
                row[k] = str(v)
        rows.append(row)

    return RankResponse(rows=rows)
