"""
database.py
-----------
SQLite storage for patients + prediction history.
"""

import os
import sqlite3
import json
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "healthcare.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        weight REAL,
        height REAL,
        created_at TEXT
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        symptoms TEXT,
        predicted_disease TEXT,
        probability REAL,
        risk_level TEXT,
        risk_score INTEGER,
        specialist TEXT,
        created_at TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    );
    """)
    conn.commit()
    conn.close()


def save_patient_and_prediction(report):
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")

    cur.execute(
        "INSERT INTO patients (name, age, gender, weight, height, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            report.get("patient_name"),
            report.get("age"),
            report.get("gender"),
            report.get("weight"),
            report.get("height"),
            now,
        ),
    )
    patient_id = cur.lastrowid

    top_disease, top_prob = (report["predictions"][0] if report.get("predictions") else (None, None))

    cur.execute(
        "INSERT INTO predictions "
        "(patient_id, symptoms, predicted_disease, probability, risk_level, risk_score, specialist, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            patient_id,
            json.dumps(report.get("symptoms", [])),
            top_disease,
            top_prob,
            report.get("risk_level"),
            report.get("risk_score"),
            report.get("specialist"),
            now,
        ),
    )
    conn.commit()
    conn.close()
    return patient_id


def get_history(limit=50):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT p.id, pt.name, pt.age, pt.gender, p.symptoms, p.predicted_disease,
               p.probability, p.risk_level, p.risk_score, p.specialist, p.created_at
        FROM predictions p
        JOIN patients pt ON pt.id = p.patient_id
        ORDER BY p.id DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()

    history = []
    for r in rows:
        history.append({
            "id": r["id"],
            "name": r["name"],
            "age": r["age"],
            "gender": r["gender"],
            "symptoms": json.loads(r["symptoms"]) if r["symptoms"] else [],
            "predicted_disease": r["predicted_disease"],
            "probability": r["probability"],
            "risk_level": r["risk_level"],
            "risk_score": r["risk_score"],
            "specialist": r["specialist"],
            "created_at": r["created_at"],
        })
    return history
