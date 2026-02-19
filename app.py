from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "ML Model is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    value = np.array([[data["value"]]])
    prediction = model.predict(value)
    return jsonify({"prediction": float(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)