"""
train_model.py
---------------
Trains a Random Forest classifier on dataset/diseases.csv (multi-hot symptom
vectors -> Disease label), evaluates it, and saves:
    model/disease_model.pkl     (trained classifier)
    model/symptom_encoder.pkl   (ordered list of symptom columns used as input)
    model/label_encoder.pkl     (sklearn LabelEncoder for disease names)
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "diseases.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    df = pd.read_csv(DATASET_PATH)
    symptom_cols = [c for c in df.columns if c != "Disease"]

    X = df[symptom_cols]
    y_raw = df["Disease"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("-" * 60)
    print(classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    ))
    print("-" * 60)
    print("Confusion matrix shape:", confusion_matrix(y_test, y_pred).shape)

    joblib.dump(model, os.path.join(MODEL_DIR, "disease_model.pkl"), compress=3)
    joblib.dump(symptom_cols, os.path.join(MODEL_DIR, "symptom_encoder.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # feature importance (nice to have, printed for transparency)
    importances = sorted(
        zip(symptom_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:10]
    print("-" * 60)
    print("Top 10 most important symptoms:")
    for name, score in importances:
        print(f"  {name:25s} {score:.4f}")

    print("=" * 60)
    print("Saved model + encoders to model/")


if __name__ == "__main__":
    main()
