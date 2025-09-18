#!/usr/bin/env python3
import argparse
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

def main():
    p = argparse.ArgumentParser(description="Train hotel cancellation risk model")
    p.add_argument("--data_path", required=True, help="CSV file with features + booking_status")
    args = p.parse_args()

    # Configuración de MLflow
    mlflow.set_tracking_uri("http://0.0.0.0:8050")
    mlflow.set_experiment("Hotel Cancellation Prediction")

    with mlflow.start_run():
        # Leer dataset
        df = pd.read_csv(args.data_path)

        # Eliminar columnas innecesarias
        for c in ["Unnamed: 0", "index"]:
            if c in df.columns:
                df = df.drop(columns=[c])

        if "booking_status" not in df.columns:
            raise ValueError("Column 'booking_status' must exist in dataset")

        # Variables
        y = df["booking_status"].astype(int)
        X = df.drop(columns=["booking_status"])

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Parámetros modelo
        params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }
        mlflow.log_params(params)

        # Entrenamiento
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluación
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        print(f"AUC: {auc:.4f}  ACC: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", acc)

        # Guarda modelo en MLflow y local
        mlflow.sklearn.log_model(model, "xgboost-model")
        #joblib.dump(model, "model.joblib")

        # Guarda metadatos de features REALES
        #meta = {"feature_cols": list(X.columns)}
        #with open("model_meta.json", "w", encoding="utf-8") as f:
         #   json.dump(meta, f, ensure_ascii=False, indent=2)
        #mlflow.log_artifact("model_meta.json")

        # Guarda modelo en carpeta api/models
        joblib.dump(model, "api/models/model.joblib")

        # Guarda metadatos en carpeta api/models
        meta = {"feature_cols": list(X.columns)}
        with open("api/models/model_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact("api/models/model_meta.json")

if __name__ == "__main__":
    main()