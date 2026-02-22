from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

# ----------------- Load dataset (label → crop mapping) -----------------
dataset_path = r"C:\Users\Dell\Desktop\Crop_recommendation.csv"
df = pd.read_csv(dataset_path)

label_to_crop = dict(enumerate(df["label"].astype("category").cat.categories))

# ----------------- Load trained model -----------------
with open("random_forest_best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------- Initialize Flask -----------------
app = Flask(__name__)
CORS(app)

# ----------------- Prediction logic -----------------
def predict_crop(input_df):
    predicted_label = model.predict(input_df)[0]
    crop_name = label_to_crop[predicted_label]
    confidence = round(np.max(model.predict_proba(input_df)) * 100, 2)
    return crop_name, confidence

# ----------------- Routes -----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Crop Recommendation API is running"})

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()

        input_df = pd.DataFrame(
            [[
                float(data["nitrogen"]),
                float(data["phosphorus"]),
                float(data["potassium"]),
                float(data["temperature"]),
                float(data["humidity"]),
                float(data["ph"]),
                float(data["rainfall"])
            ]],
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        )

        crop, confidence = predict_crop(input_df)

        return jsonify({
            "crop": crop,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "crop": "Error",
            "confidence": 0,
            "message": str(e)
        }), 500

# ----------------- Run server -----------------
if __name__ == "__main__":
    print("🚀 Starting Crop Recommendation API...")
    app.run(debug=True, port=5000)