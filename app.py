"""
app.py
------
Flask web app for the AI-Powered Healthcare Diagnosis Assistant.

IMPORTANT: This tool is for educational/informational purposes only.
It does NOT provide medical diagnoses. Always consult a qualified
healthcare professional.
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import predict
import database

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-in-production"

database.init_db()


@app.route("/", methods=["GET"])
def index():
    symptoms = predict.get_all_symptoms()
    return render_template("index.html", symptoms=symptoms)


@app.route("/predict", methods=["POST"])
def predict_route():
    form = request.form
    selected_symptoms = request.form.getlist("symptoms")

    if not selected_symptoms:
        flash("Please select at least one symptom.")
        return redirect(url_for("index"))

    patient = {
        "name": form.get("name", "").strip() or "Anonymous",
        "age": form.get("age"),
        "gender": form.get("gender"),
        "weight": form.get("weight"),
        "height": form.get("height"),
        "temperature": form.get("temperature"),
        "symptoms": selected_symptoms,
        "duration": form.get("duration"),
        "existing_diseases": form.get("existing_diseases", "").strip(),
    }

    report = predict.run_full_pipeline(patient)
    report["weight"] = patient["weight"]
    report["height"] = patient["height"]

    database.save_patient_and_prediction(report)

    return render_template("prediction.html", report=report)


@app.route("/history", methods=["GET"])
def history():
    records = database.get_history()
    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
