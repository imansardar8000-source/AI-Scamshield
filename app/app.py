import numpy as np
from flask import Flask, render_template, request
import pickle
import re

# =========================
# Load calibrated model
# =========================
with open("../model/scam_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

app = Flask(__name__)

# =========================
# Text cleaning (MUST MATCH TRAINING)
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# 🌐 Language-aware confidence
# =========================
def language_confidence_factor(text):
    # Hindi / Devanagari
    if re.search(r"[\u0900-\u097F]", text):
        return 0.9
    # Hinglish (very light penalty)
    hinglish = ["aap", "karo", "hai", "paise", "jeeto"]
    if any(w in text.lower() for w in hinglish):
        return 0.95
    return 1.0

# =========================
# 📊 Top contributing TF-IDF words
# =========================
def get_top_features(vector, vectorizer, model, top_n=5):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.calibrated_classifiers_[0].estimator.coef_[0]

    vec = vector.toarray()[0]
    contributions = vec * coef

    top_idx = np.argsort(contributions)[-top_n:][::-1]
    return feature_names[top_idx].tolist()

# =========================
# 🎯 Confidence label
# =========================
def get_confidence_label(risk):
    if risk >= 75:
        return "HIGH RISK"
    elif risk >= 40:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

# =========================
# 📉 Uncertainty warning
# =========================
def get_uncertainty_warning(risk):
    if 45 <= risk <= 55:
        return "⚠ Model confidence is borderline. Treat with caution."
    return None

# =========================
# 🧠 One-line risk explanation
# =========================
def risk_explanation(risk, prediction):
    if prediction == 1:
        if risk >= 75:
            return "High likelihood of scam due to strong scam-like language patterns."
        elif risk >= 40:
            return "Message shows mixed signals commonly seen in scam messages."
        else:
            return "Weak scam indicators detected, but caution is advised."
    return "Message language aligns with typical non-scam communication."

# =========================
# Main route
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    risk = 0
    confidence = "LOW RISK"
    scam_type = "SAFE"
    reasons = []
    top_words = []
    warning = None
    summary = ""
    model_info = "Calibrated Linear SVM | TF-IDF (1–2 grams)"

    if request.method == "POST":
        message = request.form["message"]

        cleaned = clean_text(message)
        vector = vectorizer.transform([cleaned])

        # -------- PURE ML PREDICTION --------
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0][1]

        risk = int(probability * 100)

        # 🌐 Language-aware adjustment
        risk = int(risk * language_confidence_factor(message))

        confidence = get_confidence_label(risk)
        warning = get_uncertainty_warning(risk)
        summary = risk_explanation(risk, prediction)

        if prediction == 1:
            scam_type = "SCAM LIKELY"
            reasons = [
                "Message language matches known scam distributions",
                "Model detected statistically significant scam features"
            ]
            top_words = get_top_features(vector, vectorizer, model)
        else:
            scam_type = "LIKELY SAFE"
            reasons = ["No strong scam indicators detected by the ML model"]

    return render_template(
        "index.html",
        message=message,
        risk=risk,
        scam_type=scam_type,
        confidence=confidence,
        reasons=reasons,
        top_words=top_words,
        warning=warning,
        summary=summary,
        model_info=model_info
    )

# =========================
# Run app
# =========================
if __name__ == "__main__":
    app.run(debug=True)