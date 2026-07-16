<div align="center">

# 🩺 AI-Powered Healthcare Diagnosis Assistant

**Enter symptoms → get a likely condition, a risk level, and who to see.**

An educational, end-to-end ML + web app that takes patient symptoms and returns
probable conditions, an estimated risk level, precautions, and a recommended
type of specialist — with emergency alerts for critical symptom combinations.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-history-003B57?logo=sqlite&logoColor=white)
![Status](https://img.shields.io/badge/status-educational%20demo-yellow)

</div>

> ⚠️ **This is an educational project, not a medical device.** It does not
> replace professional medical advice, diagnosis, or treatment. Always
> consult a qualified healthcare provider. The included dataset is
> **synthetically generated** — see [About the dataset](#-about-the-dataset).

---

## 📸 Screenshots

<table>
<tr>
<td width="50%">

**Symptom check** — tap-to-select symptom chips with live search & a running counter

<img width="900" height="1200" alt="01_form" src="https://github.com/user-attachments/assets/c1dc165f-6c90-4e4d-8300-87f14064166a" alt="Symptom check form" width="100%"/>


</td>
<td width="50%">

**Results report** — likely condition, risk level, confidence bars, and recommendations

<img width="900" height="1400" alt="02_report" src="https://github.com/user-attachments/assets/31506ffb-0b5c-4f25-b2c1-39aec2445daa" alt="Diagnosis report" width="100%"/>


</td>
</tr>
<tr>
<td colspan="2">

**History** — every past check, saved locally

<img width="599" height="382" alt="03_history" src="https://github.com/user-attachments/assets/e271eebe-a4e4-4336-b6be-9ad948a9d7a2" alt="Prediction history" width="100%"/>


</td>
</tr>
</table>

---

## ✨ Features

| | |
|---|---|
| 🧠 **ML-based prediction** | RandomForest classifier trained on multi-hot symptom vectors, returns top-3 conditions with confidence % |
| 🚦 **Risk scoring** | Combines age, fever, breathing symptoms, existing conditions, and severity into Low / Medium / High |
| 🚨 **Emergency detection** | Rule-based alerts for critical combinations (e.g. chest pain + breathing difficulty) |
| 👨‍⚕️ **Specialist recommendation** | Maps the predicted condition to the right type of doctor |
| 💊 **Precautions & medication info** | Informational only, always paired with a "consult a professional" disclaimer |
| 🕒 **History** | Every check is saved locally (SQLite) and viewable on the History page |
| 🖨️ **Printable report** | One click to print or save the results as a PDF |
| 📱 **Clean, responsive UI** | Custom design system, no framework bloat, works down to mobile |

---

## 🗂️ Project structure

```
AI_Healthcare_Diagnosis_Assistant/
│
├── dataset/
│   ├── diseases.csv        # symptom multi-hot vectors → Disease label (training data)
│   ├── symptoms.csv        # master symptom list + display names + severity weights
│   ├── precautions.csv     # Disease → precaution advice
│   ├── medications.csv     # Disease → informational medication notes
│   ├── doctors.csv         # Disease → recommended specialist
│   └── severity.csv        # symptom → emergency flag
│
├── model/
│   ├── disease_model.pkl       # trained RandomForestClassifier
│   ├── symptom_encoder.pkl     # ordered symptom column list used as model input
│   └── label_encoder.pkl       # sklearn LabelEncoder for disease names
│
├── images/                  # README screenshots
│
├── templates/                # Flask/Jinja HTML
│   ├── base.html
│   ├── index.html            # patient input + symptom picker
│   ├── prediction.html        # results / report page
│   └── history.html          # prediction history table
│
├── static/
│   ├── css/style.css         # custom design system
│   └── js/script.js          # chip picker + search + live counter
│
├── app.py               # Flask application + routes
├── train_model.py       # trains and evaluates the ML model
├── predict.py           # prediction pipeline (encoding, risk, recommendations)
├── database.py          # SQLite storage (patients + predictions)
├── generate_dataset.py  # generates the synthetic demo dataset (already run)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick start

```bash
cd AI_Healthcare_Diagnosis_Assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (optional — a model is already included; re-run only if you change the dataset)
python train_model.py

# run the web app
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## ⚙️ How it works

```
Patient Input → Validation → Feature Encoding → ML Disease Model
                                                        │
                                     ┌──────────────────┴──────────────────┐
                                     ▼                                     ▼
                          Disease Probabilities                 Risk Scoring Engine
                                     │                                     │
                                     └──────────────────┬──────────────────┘
                                                        ▼
                                       Recommendation Engine
                                       (Precautions, Specialist, Advice)
                                                        │
                                                        ▼
                                        Report & History Storage
```

1. **Patient input** — `index.html` collects age, gender, weight, height, temperature, duration, existing conditions, and a searchable symptom checklist.
2. **Encoding** — `predict.py:encode_symptoms()` converts selected symptoms into the multi-hot vector the model expects.
3. **Feature engineering** — BMI and a symptom severity score are computed.
4. **ML inference** — a `RandomForestClassifier` returns probabilities across all diseases; the top 3 are shown with confidence percentages.
5. **Risk engine** — `calc_risk()` combines age, fever, breathing symptoms, existing conditions, and severity into a score mapped to Low / Medium / High.
6. **Emergency detection** — `detect_emergency()` flags critical symptom combinations with an immediate warning banner.
7. **Recommendation engine** — looks up precautions, informational medication notes, and the recommended specialist for the top predicted condition.
8. **Report + history** — each check is saved to a local SQLite database (`healthcare.db`, auto-created) and viewable on the History page, with a Print/Save-as-PDF button on the report.

---

## 📊 About the dataset

`generate_dataset.py` creates a **synthetic** dataset of ~3,960 patient rows
across 18 common conditions and 38 symptoms, sampled from rough clinical
symptom-probability profiles with added noise — good enough to demo the whole
pipeline end-to-end without external downloads.

**For anything beyond a demo/portfolio project**, swap in a real, licensed
clinical dataset (e.g. a vetted Kaggle symptom-to-disease dataset) reviewed by
a medical professional, drop it in as `dataset/diseases.csv`, and re-run
`train_model.py`. Evaluation metrics (accuracy, precision, recall, F1,
confusion matrix) print automatically when you do.

**Current demo model:** ~82% accuracy on held-out synthetic data (RandomForest, 150 trees).

---

## 🧭 Roadmap / extension ideas

- 🎙️ Voice-based symptom input (speech-to-text)
- 💬 LLM-based chatbot for general health Q&A, with clear disclaimers
- 🖼️ Image-based skin condition analysis (CNN)
- 📄 PDF report generation / email / WhatsApp sharing
- 📅 Appointment booking integration
- 🌐 Multi-language UI (e.g. English, Hindi)
- 📈 Provider-facing analytics dashboard
- ☁️ Swap SQLite for MySQL/Postgres and deploy (Render, Railway, Azure, AWS)

---

## 🛡️ Ethical & safety notes (built in)

- Results are always framed as **"most likely conditions,"** never a definitive diagnosis.
- Medication information is explicitly informational, with a visible disclaimer.
- Emergency symptom combinations trigger a prominent warning banner.
- Patient data is stored **locally only** (SQLite) — no external transmission.

---

<div align="center">
<sub>Built as an educational demo of an end-to-end ML + web diagnosis pipeline. Not for clinical use.</sub>
</div>
