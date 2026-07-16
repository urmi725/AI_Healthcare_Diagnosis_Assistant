"""
generate_dataset.py
--------------------
Generates a synthetic (but medically-plausible) symptom -> disease dataset
for demo/training purposes, plus reference tables (precautions, medicines,
doctors, severity).

NOTE: This is SYNTHETIC data created for educational/demo purposes only.
For a production system, replace dataset/diseases.csv with a vetted,
licensed clinical dataset (e.g. a Kaggle disease-symptom dataset reviewed
by a medical professional).
"""

import random
import csv
import os

random.seed(42)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Master symptom list
# ---------------------------------------------------------------------------
SYMPTOMS = [
    "fever", "high_fever", "cough", "headache", "fatigue", "sore_throat",
    "runny_nose", "sneezing", "body_ache", "chills", "nausea", "vomiting",
    "diarrhea", "abdominal_pain", "chest_pain", "breathing_difficulty",
    "dizziness", "rash", "itching", "joint_pain", "loss_of_appetite",
    "weight_loss", "night_sweats", "blood_in_vomit", "blood_in_stool",
    "palpitations", "excessive_thirst", "frequent_urination",
    "blurred_vision", "numbness", "confusion", "sensitivity_to_light",
    "muscle_weakness", "yellowing_of_skin", "swelling", "sweating",
    "anxiety_feeling", "wheezing",
]

# ---------------------------------------------------------------------------
# 2. Disease -> {symptom: probability} profiles
#    (This encodes rough clinical intuition, NOT verified medical fact.)
# ---------------------------------------------------------------------------
DISEASE_PROFILES = {
    "Flu": {
        "fever": 0.9, "high_fever": 0.4, "cough": 0.7, "headache": 0.6,
        "fatigue": 0.85, "body_ache": 0.8, "chills": 0.7, "sore_throat": 0.4,
        "loss_of_appetite": 0.4,
    },
    "Common Cold": {
        "runny_nose": 0.9, "sneezing": 0.85, "sore_throat": 0.7,
        "cough": 0.6, "fever": 0.2, "headache": 0.3, "fatigue": 0.4,
    },
    "COVID-19": {
        "fever": 0.8, "cough": 0.8, "fatigue": 0.7, "breathing_difficulty": 0.5,
        "sore_throat": 0.4, "loss_of_appetite": 0.3, "body_ache": 0.5,
        "headache": 0.4,
    },
    "Food Poisoning": {
        "vomiting": 0.85, "diarrhea": 0.85, "abdominal_pain": 0.8,
        "nausea": 0.8, "fever": 0.3, "fatigue": 0.4,
    },
    "Migraine": {
        "headache": 0.95, "sensitivity_to_light": 0.7, "nausea": 0.5,
        "dizziness": 0.4, "fatigue": 0.3,
    },
    "Dengue": {
        "high_fever": 0.9, "body_ache": 0.85, "joint_pain": 0.8,
        "rash": 0.5, "fatigue": 0.6, "vomiting": 0.3, "headache": 0.5,
        "blood_in_vomit": 0.05,
    },
    "Typhoid": {
        "fever": 0.9, "high_fever": 0.4, "abdominal_pain": 0.6,
        "loss_of_appetite": 0.6, "fatigue": 0.6, "headache": 0.4,
        "diarrhea": 0.3,
    },
    "Gastritis": {
        "abdominal_pain": 0.85, "nausea": 0.6, "vomiting": 0.4,
        "loss_of_appetite": 0.5, "bloating": 0.0,
    },
    "Bronchitis": {
        "cough": 0.9, "fatigue": 0.5, "chest_pain": 0.3,
        "breathing_difficulty": 0.4, "wheezing": 0.5, "fever": 0.3,
    },
    "Pneumonia": {
        "fever": 0.8, "high_fever": 0.5, "cough": 0.85,
        "breathing_difficulty": 0.7, "chest_pain": 0.6, "fatigue": 0.6,
        "confusion": 0.1,
    },
    "Diabetes": {
        "excessive_thirst": 0.85, "frequent_urination": 0.85,
        "fatigue": 0.6, "blurred_vision": 0.5, "weight_loss": 0.4,
        "numbness": 0.2,
    },
    "Hypertension": {
        "headache": 0.5, "dizziness": 0.5, "blurred_vision": 0.3,
        "chest_pain": 0.2, "palpitations": 0.3, "fatigue": 0.3,
    },
    "Allergic Rhinitis": {
        "sneezing": 0.9, "runny_nose": 0.85, "itching": 0.6,
        "sore_throat": 0.2, "swelling": 0.2,
    },
    "Chickenpox": {
        "rash": 0.9, "itching": 0.8, "fever": 0.6, "fatigue": 0.5,
        "loss_of_appetite": 0.3,
    },
    "Jaundice": {
        "yellowing_of_skin": 0.9, "fatigue": 0.6, "loss_of_appetite": 0.6,
        "abdominal_pain": 0.4, "nausea": 0.4, "vomiting": 0.2,
    },
    "Urinary Tract Infection": {
        "frequent_urination": 0.8, "abdominal_pain": 0.5, "fever": 0.3,
        "fatigue": 0.3,
    },
    "Anxiety Disorder": {
        "anxiety_feeling": 0.9, "palpitations": 0.6, "dizziness": 0.4,
        "fatigue": 0.5, "sweating": 0.5, "breathing_difficulty": 0.3,
    },
    "Asthma": {
        "breathing_difficulty": 0.85, "wheezing": 0.8, "cough": 0.6,
        "chest_pain": 0.3, "fatigue": 0.3,
    },
}

DISEASES = list(DISEASE_PROFILES.keys())

# ---------------------------------------------------------------------------
# 3. Generate synthetic patient rows by sampling from each disease profile
# ---------------------------------------------------------------------------
ROWS_PER_DISEASE = 220
NOISE_SYMPTOM_PROB = 0.03  # small chance an unrelated symptom is also present

rows = []
for disease, profile in DISEASE_PROFILES.items():
    for _ in range(ROWS_PER_DISEASE):
        row = {s: 0 for s in SYMPTOMS}
        for symptom, prob in profile.items():
            if symptom in row and random.random() < prob:
                row[symptom] = 1
        # small amount of random noise so the model doesn't overfit to exact patterns
        for s in SYMPTOMS:
            if row[s] == 0 and random.random() < NOISE_SYMPTOM_PROB:
                row[s] = 1
        # ensure at least one symptom present
        if sum(row.values()) == 0:
            row[random.choice(list(profile.keys()))] = 1
        row["Disease"] = disease
        rows.append(row)

random.shuffle(rows)

with open(os.path.join(DATASET_DIR, "diseases.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=SYMPTOMS + ["Disease"])
    writer.writeheader()
    writer.writerows(rows)

print(f"diseases.csv written: {len(rows)} rows, {len(SYMPTOMS)} symptoms, {len(DISEASES)} diseases")

# ---------------------------------------------------------------------------
# 4. symptoms.csv  (symptom, display_name, base_severity 1-5)
# ---------------------------------------------------------------------------
SEVERITY_MAP = {
    "fever": 3, "high_fever": 4, "cough": 2, "headache": 2, "fatigue": 2,
    "sore_throat": 1, "runny_nose": 1, "sneezing": 1, "body_ache": 2,
    "chills": 2, "nausea": 2, "vomiting": 3, "diarrhea": 3,
    "abdominal_pain": 3, "chest_pain": 5, "breathing_difficulty": 5,
    "dizziness": 3, "rash": 2, "itching": 1, "joint_pain": 2,
    "loss_of_appetite": 2, "weight_loss": 3, "night_sweats": 3,
    "blood_in_vomit": 5, "blood_in_stool": 5, "palpitations": 4,
    "excessive_thirst": 2, "frequent_urination": 2, "blurred_vision": 3,
    "numbness": 3, "confusion": 5, "sensitivity_to_light": 2,
    "muscle_weakness": 3, "yellowing_of_skin": 4, "swelling": 3,
    "sweating": 1, "anxiety_feeling": 2, "wheezing": 4,
}

with open(os.path.join(DATASET_DIR, "symptoms.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["symptom", "display_name", "severity_weight"])
    for s in SYMPTOMS:
        writer.writerow([s, s.replace("_", " ").title(), SEVERITY_MAP.get(s, 2)])

# ---------------------------------------------------------------------------
# 5. severity.csv (symptom -> emergency flag)
# ---------------------------------------------------------------------------
EMERGENCY_SYMPTOMS = {
    "chest_pain", "breathing_difficulty", "blood_in_vomit", "blood_in_stool",
    "confusion",
}
with open(os.path.join(DATASET_DIR, "severity.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["symptom", "is_emergency_flag"])
    for s in SYMPTOMS:
        writer.writerow([s, 1 if s in EMERGENCY_SYMPTOMS else 0])

# ---------------------------------------------------------------------------
# 6. precautions.csv
# ---------------------------------------------------------------------------
PRECAUTIONS = {
    "Flu": ["Rest and hydrate", "Isolate to avoid spreading", "Monitor temperature", "Consult a physician if symptoms worsen"],
    "Common Cold": ["Rest", "Drink warm fluids", "Steam inhalation", "Avoid cold exposure"],
    "COVID-19": ["Isolate immediately", "Monitor oxygen levels", "Stay hydrated", "Seek care if breathing difficulty occurs"],
    "Food Poisoning": ["Stay hydrated (ORS)", "Avoid solid food temporarily", "Rest", "Seek care if symptoms persist beyond 48 hrs"],
    "Migraine": ["Rest in a dark quiet room", "Stay hydrated", "Avoid known triggers", "Consult neurologist if frequent"],
    "Dengue": ["Stay hydrated", "Avoid aspirin/ibuprofen", "Monitor platelet count", "Seek immediate care if bleeding occurs"],
    "Typhoid": ["Drink boiled/purified water", "Maintain hygiene", "Complete full antibiotic course if prescribed", "Rest"],
    "Gastritis": ["Avoid spicy/oily food", "Eat smaller meals", "Avoid alcohol", "Consult doctor if pain persists"],
    "Bronchitis": ["Rest voice and body", "Stay hydrated", "Avoid smoke/irritants", "Use humidifier"],
    "Pneumonia": ["Seek prompt medical evaluation", "Rest", "Stay hydrated", "Monitor breathing closely"],
    "Diabetes": ["Monitor blood sugar", "Maintain balanced diet", "Regular exercise", "Consult endocrinologist"],
    "Hypertension": ["Reduce salt intake", "Monitor blood pressure regularly", "Manage stress", "Consult physician"],
    "Allergic Rhinitis": ["Avoid known allergens", "Keep living areas dust-free", "Consider antihistamines (consult doctor)"],
    "Chickenpox": ["Isolate to prevent spread", "Avoid scratching", "Keep skin clean", "Consult doctor for children/adults at risk"],
    "Jaundice": ["Rest", "Avoid alcohol and fatty food", "Stay hydrated", "Seek prompt medical evaluation"],
    "Urinary Tract Infection": ["Drink plenty of water", "Maintain hygiene", "Avoid holding urine", "Consult doctor for antibiotics"],
    "Anxiety Disorder": ["Practice relaxation/breathing exercises", "Reduce caffeine", "Maintain sleep routine", "Consider consulting a mental health professional"],
    "Asthma": ["Avoid triggers (dust, smoke)", "Keep rescue inhaler accessible", "Monitor breathing", "Seek care if attacks worsen"],
}
with open(os.path.join(DATASET_DIR, "precautions.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Disease", "Precaution"])
    for disease, plist in PRECAUTIONS.items():
        for p in plist:
            writer.writerow([disease, p])

# ---------------------------------------------------------------------------
# 7. medications.csv (informational only)
# ---------------------------------------------------------------------------
MEDICATIONS = {
    "Flu": "Paracetamol (fever/pain), fluids, rest",
    "Common Cold": "Cetirizine (antihistamine), Paracetamol if needed",
    "COVID-19": "Paracetamol for fever; follow physician guidance",
    "Food Poisoning": "Oral Rehydration Salts (ORS)",
    "Migraine": "Paracetamol/Ibuprofen (consult doctor for frequent migraines)",
    "Dengue": "Paracetamol only (AVOID Aspirin/Ibuprofen)",
    "Typhoid": "Antibiotics as prescribed by physician",
    "Gastritis": "Antacid",
    "Bronchitis": "Cough suppressant, fluids",
    "Pneumonia": "Antibiotics as prescribed (needs medical evaluation)",
    "Diabetes": "As prescribed by endocrinologist (e.g., Metformin)",
    "Hypertension": "As prescribed by physician (e.g., ACE inhibitors)",
    "Allergic Rhinitis": "Antihistamine (e.g., Cetirizine)",
    "Chickenpox": "Calamine lotion, antihistamine for itching",
    "Jaundice": "Supportive care; needs medical evaluation",
    "Urinary Tract Infection": "Antibiotics as prescribed by physician",
    "Anxiety Disorder": "Consult mental health professional; therapy first-line",
    "Asthma": "Bronchodilator inhaler as prescribed",
}
with open(os.path.join(DATASET_DIR, "medications.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Disease", "Medication_Info"])
    for disease, med in MEDICATIONS.items():
        writer.writerow([disease, med])

# ---------------------------------------------------------------------------
# 8. doctors.csv (disease -> specialist)
# ---------------------------------------------------------------------------
SPECIALISTS = {
    "Flu": "General Physician",
    "Common Cold": "General Physician",
    "COVID-19": "General Physician / Pulmonologist",
    "Food Poisoning": "General Physician / Gastroenterologist",
    "Migraine": "Neurologist",
    "Dengue": "General Physician / Infectious Disease Specialist",
    "Typhoid": "General Physician / Infectious Disease Specialist",
    "Gastritis": "Gastroenterologist",
    "Bronchitis": "Pulmonologist",
    "Pneumonia": "Pulmonologist",
    "Diabetes": "Endocrinologist",
    "Hypertension": "Cardiologist",
    "Allergic Rhinitis": "ENT Specialist / Allergist",
    "Chickenpox": "Dermatologist / General Physician",
    "Jaundice": "Hepatologist / Gastroenterologist",
    "Urinary Tract Infection": "Urologist",
    "Anxiety Disorder": "Psychiatrist / Psychologist",
    "Asthma": "Pulmonologist",
}
with open(os.path.join(DATASET_DIR, "doctors.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Disease", "Recommended_Specialist"])
    for disease, spec in SPECIALISTS.items():
        writer.writerow([disease, spec])

print("All reference CSVs written to dataset/")
