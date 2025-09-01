#!/usr/bin/env python3
import argparse
import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from xgboost import XGBClassifier

def main():
    p = argparse.ArgumentParser(description="Train hotel cancellation risk model")
    p.add_argument("--data_path", required=True, help="CSV file with features + booking_status")
    p.add_argument("--model_path", default="model.joblib", help="Where to write the trained model")
    p.add_argument("--meta_path", default="model_meta.json", help="Where to write model metadata")
    args = p.parse_args()

    df = pd.read_csv(args.data_path)
    # Drop obvious index cols if present
    for c in ["Unnamed: 0", "index"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    if "booking_status" not in df.columns:
        raise ValueError("Column 'booking_status' must exist in dataset")

    y = df["booking_status"].astype(int)
    X = df.drop(columns=["booking_status"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Simple, strong baseline
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # Metrics
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    print(f"AUC: {auc:.4f}  ACC: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model + feature names
    joblib.dump(model, args.model_path)
    meta = {"feature_names": list(X.columns)}
    with open(args.meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
