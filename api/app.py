from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import json
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Montar carpeta estática
#app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Cargar modelo
model = joblib.load(BASE_DIR / "models/model.joblib")

# Cargar metadatos
with open(BASE_DIR / "models/model_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
feature_cols = meta["feature_cols"]

app = FastAPI(title="Hotel Cancellation Prediction API")

# Montar carpeta estática para servir HTML, CSS, JS
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Pydantic model dinámico (cualquier dict con features)
class BookingFeatures(BaseModel):
    data: dict

# --- Diccionario con min y max reales (según tu EDA) ---
scales = {
    "no_of_adults": (0, 4),
    "no_of_children": (0, 9),
    "no_of_weekend_nights": (0, 7),
    "no_of_week_nights": (0, 17),
    "type_of_meal_plan": (0, 3),
    "required_car_parking_space": (0, 1),
    "room_type_reserved": (0, 6),
    "lead_time": (0, 443),
    "arrival_year": (2017, 2018),
    "arrival_month": (1, 12),
    "arrival_date": (1, 31),
    "market_segment_type": (0, 4),
    "repeated_guest": (0, 1),
    "no_of_previous_cancellations": (0, 13),
    "no_of_previous_bookings_not_canceled": (0, 58),
    "avg_price_per_room": (0, 540),
    "no_of_special_requests": (0, 5),
}

def desnormalizar(fila: dict) -> dict:
    """Convierte de valores normalizados (0-1) a originales usando min/max."""
    desnorm = {}
    for col, val in fila.items():
        if col in scales:
            minv, maxv = scales[col]
            desnorm[col] = round(val * (maxv - minv) + minv, 2)
        else:
            desnorm[col] = val
    return desnorm

# Servir la página principal
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    with open(BASE_DIR / "static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
def predict(features: BookingFeatures):
    # Convertir a DataFrame
    df = pd.DataFrame([features.data])

    # Asegurar todas las columnas en el orden correcto
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0  # Si falta, poner en 0

    df = df[feature_cols]

    # Predicción
    proba = model.predict_proba(df)[:, 1][0]
    pred = int(proba >= 0.5)

    # Crear versión desnormalizada para mostrar en el historial
    desnorm_data = desnormalizar(features.data)

    return {
        "prediction": pred,
        "cancellation_probability": float(proba),
        "inputs_original": desnorm_data  # 🔹 se envían valores originales al frontend
    }