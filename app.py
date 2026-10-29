"""Flask API for ML-based weather prediction. No HTML templates are used."""
from flask import Flask, jsonify, request
from model import predict_weather, ModelNotTrainedError

app = Flask(__name__)

@app.get("/")
def index():
    return jsonify({
        "name": "Weather Prediction API",
        "status": "ok",
        "usage": "POST /predict with JSON containing 7-10 days of historical weather data.",
        "example": {
            "history": [
                {"temperature": 24.5, "humidity": 70, "wind_speed": 8.2, "condition": "Cloudy"}
            ] * 7
        }
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    history = payload.get("history")
    if not isinstance(history, list):
        return jsonify({"error": "'history' must be a list of 7-10 daily records"}), 400
    try:
        result = predict_weather(history)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except ModelNotTrainedError as exc:
        return jsonify({"error": str(exc), "hint": "Run `python train.py` first."}), 503
    return jsonify(result)

@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == "__main__":
    # For local development only. Use a production WSGI server for deployment.
    app.run(host="127.0.0.1", port=5000, debug=False)
