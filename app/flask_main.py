from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os

# Initialize app
app = Flask(__name__)

# Load trained model
model = joblib.load('models/battery_health_model.pkl')

# Serve the HTML dashboard
@app.route("/", methods=["GET"])
def serve_dashboard():
    dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dashboard')
    return send_from_directory(dashboard_path, 'index.html')

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Validate input
    required_fields = ["battery_voltage", "motor_temperature", "brake_pressure", "tire_pressure"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing fields in input."}), 400
    # Convert input to dataframe
    input_df = pd.DataFrame([data])
    # Predict
    prediction = model.predict(input_df)[0]
    return jsonify({
        "prediction": int(prediction),
        "status": "FAIL" if prediction == 1 else "OK"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True) 