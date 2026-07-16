"""
predict.py
----------
End-to-end prediction pipeline:
  user symptoms -> encoding -> ML model -> top-N diseases + probabilities
  -> risk scoring -> emergency detection -> recommendation engine -> report dict
"""

import os
import csv
import joblib

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

_model = None
_symptom_cols = None
_label_encoder = None


def _load_artifacts():
    global _model, _symptom_cols, _label_encoder
    if _model is None:
        _model = joblib.load(os.path.join(MODEL_DIR, "disease_model.pkl"))
        _symptom_cols = joblib.load(os.path.join(MODEL_DIR, "symptom_encoder.pkl"))
        _label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return _model, _symptom_cols, _label_encoder


def _load_csv_dict(filename, key_col, val_col, multi=False):
    path = os.path.join(DATASET_DIR, filename)
    result = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k, v = row[key_col], row[val_col]
            if multi:
                result.setdefault(k, []).append(v)
            else:
                result[k] = v
    return result


def get_all_symptoms():
    """Returns list of (symptom_key, display_name) for building the input form."""
    path = os.path.join(DATASET_DIR, "symptoms.csv")
    out = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append((row["symptom"], row["display_name"]))
    return out


def _severity_weights():
    path = os.path.join(DATASET_DIR, "symptoms.csv")
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["symptom"]] = int(row["severity_weight"])
    return out


def _emergency_flags():
    path = os.path.join(DATASET_DIR, "severity.csv")
    out = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["is_emergency_flag"] == "1":
                out.add(row["symptom"])
    return out


def encode_symptoms(selected_symptoms, symptom_cols):
    """Multi-hot encode a list of symptom keys into the model's expected column order."""
    selected = set(selected_symptoms)
    return [1 if s in selected else 0 for s in symptom_cols]


def calc_bmi(weight_kg, height_m):
    if not weight_kg or not height_m:
        return None
    try:
        return round(float(weight_kg) / (float(height_m) ** 2), 1)
    except (ValueError, ZeroDivisionError):
        return None


def calc_severity_score(selected_symptoms):
    weights = _severity_weights()
    return sum(weights.get(s, 1) for s in selected_symptoms)


def calc_risk(age, selected_symptoms, existing_diseases, temperature_c, severity_score):
    """Simple additive risk scoring engine -> (score, level)."""
    score = 0

    # Age factor
    try:
        age = int(age)
    except (TypeError, ValueError):
        age = 0
    if age >= 65:
        score += 20
    elif age >= 45:
        score += 10
    elif age <= 5:
        score += 15  # infants/young children also higher risk

    # Severity (symptom count/weight based)
    score += min(severity_score * 2, 40)

    # High fever
    try:
        temp = float(temperature_c) if temperature_c not in (None, "") else None
    except ValueError:
        temp = None
    if temp is not None and temp >= 39.0:
        score += 25
    elif temp is not None and temp >= 38.0:
        score += 10

    # Breathing issues
    if "breathing_difficulty" in selected_symptoms or "wheezing" in selected_symptoms:
        score += 40

    # Existing chronic disease
    if existing_diseases:
        score += 20

    # Emergency symptoms add heavily
    emergency = _emergency_flags()
    for s in selected_symptoms:
        if s in emergency:
            score += 15

    if score <= 30:
        level = "Low"
    elif score <= 60:
        level = "Medium"
    else:
        level = "High"

    return score, level


def detect_emergency(selected_symptoms):
    """Rule-based emergency detection for critical symptom combinations."""
    s = set(selected_symptoms)
    alerts = []
    if "chest_pain" in s and "breathing_difficulty" in s:
        alerts.append("Chest pain with breathing difficulty — seek EMERGENCY care immediately.")
    if "blood_in_vomit" in s:
        alerts.append("Blood in vomit — seek EMERGENCY care immediately.")
    if "blood_in_stool" in s:
        alerts.append("Blood in stool — seek EMERGENCY care immediately.")
    if "confusion" in s and ("high_fever" in s or "fever" in s):
        alerts.append("Confusion with fever — seek urgent medical evaluation.")
    if "breathing_difficulty" in s and "yellowing_of_skin" in s:
        alerts.append("Breathing difficulty with jaundice — seek urgent medical evaluation.")
    return alerts


def predict_diseases(selected_symptoms, top_n=3):
    """Returns list of (disease_name, probability_percent) sorted descending."""
    import pandas as pd
    model, symptom_cols, label_encoder = _load_artifacts()
    vector = encode_symptoms(selected_symptoms, symptom_cols)
    X = pd.DataFrame([vector], columns=symptom_cols)
    proba = model.predict_proba(X)[0]
    diseases = label_encoder.classes_
    ranked = sorted(zip(diseases, proba), key=lambda x: x[1], reverse=True)[:top_n]
    return [(name, round(p * 100, 1)) for name, p in ranked]


def get_recommendations(disease_name):
    """Look up precautions, medication info, and specialist for a disease."""
    precautions = _load_csv_dict("precautions.csv", "Disease", "Precaution", multi=True)
    medications = _load_csv_dict("medications.csv", "Disease", "Medication_Info")
    doctors = _load_csv_dict("doctors.csv", "Disease", "Recommended_Specialist")

    return {
        "precautions": precautions.get(disease_name, ["Consult a healthcare professional."]),
        "medication_info": medications.get(disease_name, "Consult a healthcare professional."),
        "specialist": doctors.get(disease_name, "General Physician"),
    }


def run_full_pipeline(patient):
    """
    patient: dict with keys
        name, age, gender, weight, height, temperature,
        symptoms (list), duration, existing_diseases
    Returns a full report dict.
    """
    selected_symptoms = patient.get("symptoms", [])
    severity_score = calc_severity_score(selected_symptoms)
    bmi = calc_bmi(patient.get("weight"), patient.get("height"))
    risk_score, risk_level = calc_risk(
        patient.get("age"), selected_symptoms, patient.get("existing_diseases"),
        patient.get("temperature"), severity_score,
    )
    emergency_alerts = detect_emergency(selected_symptoms)
    predictions = predict_diseases(selected_symptoms, top_n=3)

    top_disease = predictions[0][0] if predictions else None
    recs = get_recommendations(top_disease) if top_disease else {}

    report = {
        "patient_name": patient.get("name", "N/A"),
        "age": patient.get("age"),
        "gender": patient.get("gender"),
        "bmi": bmi,
        "symptoms": selected_symptoms,
        "duration": patient.get("duration"),
        "existing_diseases": patient.get("existing_diseases"),
        "severity_score": severity_score,
        "predictions": predictions,          # [(disease, prob%), ...]
        "top_disease": top_disease,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "emergency_alerts": emergency_alerts,
        "precautions": recs.get("precautions", []),
        "medication_info": recs.get("medication_info", ""),
        "specialist": recs.get("specialist", ""),
    }
    return report
