from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)   # allows React frontend to connect

# Load trained model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Root route (test route)
@app.route("/")
def home():
    return "Crop Recommendation API is running"

# Prediction API route
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # Get values from frontend
        nitrogen = float(data["nitrogen"])
        phosphorus = float(data["phosphorus"])
        potassium = float(data["potassium"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])
        rainfall = float(data["rainfall"])

        # Arrange input for model
        input_data = np.array([[nitrogen, phosphorus, potassium,
                                temperature, humidity, ph, rainfall]])

        # Predict crop
        prediction = model.predict(input_data)[0]

        # Optional confidence score (if model supports it)
        confidence = 0
        if hasattr(model, "predict_proba"):
            confidence = round(np.max(model.predict_proba(input_data)) * 100, 2)

        # Send result to frontend
        return jsonify({
            "success": True,
            "crop": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


# Run server
if __name__ == "__main__":
    print("Starting Flask API...")
    app.run(debug=True, port=5000)
