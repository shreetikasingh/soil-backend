from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

# ----------------- Load dataset and extract unique crops -----------------
dataset_path = r"C:\Users\Dell\Desktop\Crop_recommendation.csv"  # Update path if needed
df = pd.read_csv(dataset_path)
unique_crops = df['label'].unique()
crop_mapping = {i: crop for i, crop in enumerate(unique_crops)}

# ----------------- Load trained model -----------------
with open("random_forest_best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------- Initialize Flask -----------------
app = Flask(__name__)
CORS(app)

# ----------------- Flask Routes -----------------
@app.route("/")
def home():
    return "Crop Recommendation API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]
        input_data = np.array([[float(data[f]) for f in features]])

        prediction_label = model.predict(input_data)[0]
        prediction_name = crop_mapping.get(prediction_label, "Unknown")

        confidence = 0
        if hasattr(model, "predict_proba"):
            confidence = round(np.max(model.predict_proba(input_data)) * 100, 2)

        return jsonify({
            "success": True,
            "crop": prediction_name,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ----------------- Terminal Testing -----------------
def terminal_predict():
    print("=== Terminal Crop Prediction ===")
    try:
        features = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "Soil pH", "Rainfall"]
        input_values = [float(input(f"Enter {f}: ")) for f in features]

        input_data = np.array([input_values])
        prediction_label = model.predict(input_data)[0]
        prediction_name = crop_mapping.get(prediction_label, "Unknown")

        confidence = 0
        if hasattr(model, "predict_proba"):
            confidence = round(np.max(model.predict_proba(input_data)) * 100, 2)

        print(f"\nPredicted Crop: {prediction_name}")
        print(f"Confidence: {confidence}%")

    except Exception as e:
        print(f"Error: {e}")

# ----------------- Run Script -----------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].lower() == "terminal":
        terminal_predict()
    else:
        print("Starting Flask API on http://127.0.0.1:5000")
        app.run(debug=True, port=5000)
