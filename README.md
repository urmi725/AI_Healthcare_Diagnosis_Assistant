# AI-Powered Healthcare Diagnosis Assistant

An educational, end-to-end demo system that takes patient symptoms and returns:

- The most probable condition(s), as likely possibilities (not a definitive diagnosis)
- An estimated risk level (Low / Medium / High)
- A recommended type of medical specialist
- General precautions and informational medication notes
- Emergency alerts for critical symptom combinations

> ⚠️ **This is an educational project, not a medical device.** It does not
> replace professional medical advice, diagnosis, or treatment. Always
> consult a qualified healthcare provider. The included dataset is
> **synthetically generated** for demo purposes — see "About the dataset" below.

## Project structure

```
AI_Healthcare_Diagnosis_Assistant/
│
├── dataset/
│   ├── diseases.csv        # symptom multi-hot vectors -> Disease label (training data)
│   ├── symptoms.csv        # master symptom list + display names + severity weights
│   ├── precautions.csv     # Disease -> precaution advice
│   ├── medications.csv     # Disease -> informational medication notes
│   ├── doctors.csv         # Disease -> recommended specialist
│   └── severity.csv        # symptom -> emergency flag
│
├── model/
│   ├── disease_model.pkl       # trained RandomForestClassifier
│   ├── symptom_encoder.pkl     # ordered symptom column list used as model input
│   └── label_encoder.pkl       # sklearn LabelEncoder for disease names
│
├── templates/               # Flask/Jinja HTML (Bootstrap 5)
│   ├── base.html
│   ├── index.html            # patient input form
│   ├── prediction.html       # report / results page
│   └── history.html          # prediction history table
│
├── static/
│   ├── css/style.css
│   └── js/script.js
│
├── app.py               # Flask application + routes
├── train_model.py       # trains and evaluates the ML model
├── predict.py           # prediction pipeline (encoding, risk, recommendations)
├── database.py          # SQLite storage (patients + predictions)
├── generate_dataset.py  # generates the synthetic demo dataset (already run)
├── requirements.txt
└── README.md
```

## Quick start

```bash
cd AI_Healthcare_Diagnosis_Assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (Optional — a model is already included; re-run only if you change the dataset)
python train_model.py

# Run the web app
python app.py
```

Then open **http://localhost:5000** in your browser.

## How it works

1. **Patient Input** — `index.html` collects age, gender, weight, height,
   temperature, duration, existing conditions, and a searchable checklist of symptoms.
2. **Encoding** — `predict.py:encode_symptoms()` converts the selected symptoms
   into the multi-hot vector the model expects.
3. **Feature engineering** — BMI and a symptom severity score are computed.
4. **ML inference** — a `RandomForestClassifier` (trained in `train_model.py`)
   returns probabilities across all diseases; the top 3 are shown with
   confidence percentages.
5. **Risk engine** — `calc_risk()` combines age, fever, breathing symptoms,
   existing conditions, and severity into a 0+ score mapped to Low/Medium/High.
6. **Emergency detection** — `detect_emergency()` flags critical symptom
   combinations (e.g. chest pain + breathing difficulty) with an immediate warning banner.
7. **Recommendation engine** — looks up precautions, informational medication
   notes, and the recommended specialist type for the top predicted condition.
8. **Report + history** — each check is saved to a local SQLite database
   (`healthcare.db`, auto-created) and viewable on the History page. The report
   page has a Print/Save-as-PDF button.

## About the dataset

`generate_dataset.py` creates a **synthetic** dataset of ~3,960 patient rows
across 18 common conditions and 38 symptoms, sampled from rough clinical
symptom-probability profiles with added noise. This lets the whole pipeline
run end-to-end without needing external downloads.

**For anything beyond a demo/portfolio project**, replace `dataset/diseases.csv`
with a real, licensed clinical dataset (e.g. a vetted Kaggle
symptom-to-disease dataset) reviewed by a medical professional, and re-run
`train_model.py`. Model evaluation metrics (accuracy, precision, recall,
F1, classification report) print automatically when you do.

## Extending this project (ideas from the original spec)

- Voice-based symptom input (speech-to-text)
- LLM-based chatbot for general health Q&A, with clear disclaimers
- Image-based skin condition analysis (CNN)
- PDF report generation / email / WhatsApp sharing
- Appointment booking integration
- Multi-language UI (e.g. English, Hindi)
- Provider-facing analytics dashboard
- Swap SQLite for MySQL/Postgres and deploy (Render, Railway, Azure, AWS)

## Ethical & safety notes (built in)

- All results are framed as "most likely conditions," never a definitive diagnosis.
- Medication information is explicitly informational, with a visible disclaimer.
- Emergency symptom combinations trigger a prominent warning banner.
- Patient data is stored locally only (SQLite) — no external transmission.

## Screenshot of the Project 

- Form screenshot
<img width="900" height="1200" alt="01_form" src="https://github.com/user-attachments/assets/c4f51fad-33f3-495c-80f6-bd81c2b3c45c" />

-Report screenshot
  <img width="900" height="1400" alt="02_report" src="https://github.com/user-attachments/assets/002dd931-7af7-4207-a5fc-cd5d9e4ca082" />

- History screenshot
<img width="599" height="382" alt="03_history" src="https://github.com/user-attachments/assets/9b28fb22-2ddb-4415-b983-661089c49665" />

