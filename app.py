from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import sqlite3
import json
from datetime import datetime

#--------------------database---------------------------------
def save_log(disease, inputs, prediction, confidence):
    # convert numpy arrays to lists
    if isinstance(inputs, np.ndarray):
        inputs = inputs.tolist()

    conn = sqlite3.connect("prediction_logs.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO logs(timestamp, disease, inputs, prediction, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        disease,
        json.dumps(inputs),
        prediction,
        float(confidence)
    ))

    conn.commit()
    conn.close()


#===========================

app = Flask(__name__)


# Load models
diabetes_model, diabetes_scaler = pickle.load(open("models/diabetes_model.pkl","rb"))
heart_model, heart_scaler = pickle.load(open("models/heart_model.pkl","rb"))
kidney_model, kidney_scaler = pickle.load(open("models/kidney_model.pkl","rb"))

def classify(p):
    if p < 0.33: return "Low Risk"
    if p < 0.66: return "Medium Risk"
    return "High Risk"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    disease = request.form["disease"]

    # ---------------- DIABETES -----------------------------------
    if disease == "diabetes":
        feats = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        feats = np.array(feats).reshape(1, -1)
        p = diabetes_model.predict_proba(diabetes_scaler.transform(feats))[0][1]
        risk = classify(p)

        # save to database
        save_log("diabetes", feats, risk, p)

        #return f"Diabetes Prediction: {risk} (Confidence: {p:.4f})"
        return jsonify({
            "result": f"Diabetes Prediction: {risk}",
            "prob": float(p)
        })

    # --------------- HEART ---------------------
    if disease == "heart":
        feats = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['dataset']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalch']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        feats = np.array(feats).reshape(1, -1)
        p = heart_model.predict_proba(heart_scaler.transform(feats))[0][1]
        risk = classify(p)

        # save to database
        save_log("heart", feats, risk, p)

        #return f"Heart Disease Prediction: {risk} (Confidence: {p:.4f})"
        return jsonify({
            "result": f"Heart Disease Prediction: {risk}",
            "prob": float(p)
        })

    # -------------------- KIDNEY --------------------------
    if disease == "kidney":
        kidney_cols = [
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
            'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
            'appet', 'pe', 'ane'
        ]

        feats = []
        for col in kidney_cols:
            raw = request.form.get(col)
            print(col, "=>", raw)  # DEBUG

            if raw is None or raw.strip() == "":
                raw = "0"

            feats.append(float(raw))

        feats = np.array(feats).reshape(1, -1)

        feats = np.array(feats).reshape(1, -1)
        p = kidney_model.predict_proba(kidney_scaler.transform(feats))[0][1]
        risk = classify(p)

        # save to database
        save_log("kidney", feats, risk, p)

        #return f"Kidney Disease Prediction: {risk} (Confidence: {p:.4f})"
        return jsonify({
            "result": f"Kidney Disease Prediction: {risk}",
            "prob": float(p)
        })

    #return "Invalid request"
    return jsonify({"result": "Invalid request", "prob": 0})


# logs ROUTE

@app.route('/logs_table')
def logs_table():
    conn = sqlite3.connect("prediction_logs.db")
    c = conn.cursor()
    c.execute("SELECT * FROM logs ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    # convert rows into dicts for easy use in template
    logs = []
    for r in rows:
        logs.append({
            "id": r[0],
            "timestamp": r[1],
            "disease": r[2],
            "inputs": r[3],
            "prediction": r[4],
            "confidence": round(float(r[5]), 4)
        })

    return render_template("logs.html", logs=logs)



if __name__ == "__main__":
    app.run(debug=True)

