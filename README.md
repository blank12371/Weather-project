# Weather Prediction API (Flask + scikit-learn)

A small Python-only Flask API that accepts **7–10 days** of historical daily weather observations and returns:
- a predicted next-day category: `Sunny`, `Rainy`, or `Cloudy`
- a predicted next-day temperature in °C
- classifier probability scores and top-class confidence

No HTML templates or separate frontend are used. Requests and responses are JSON.

> **Important limitation:** The included training script creates synthetic data so the project runs without downloading a dataset or using an external weather service. The resulting model is a demonstration, not a scientifically validated forecasting system. Do not use it for safety-critical decisions. For real forecasts, train on quality-controlled local historical observations and validate on chronological holdout periods.

## Project layout

```text
weather_prediction_flask/
├── app.py
├── model.py
├── train.py
├── requirements.txt
├── README.md
└── models/
    ├── weather_classifier.joblib   # created by `python train.py`
    └── temperature_regressor.joblib
```

The trained model files are generated locally by the training script rather than bundled as opaque binaries. This keeps the model artifacts reproducible and makes it clear how they were trained.

## 1. Requirements

- Python 3.10 or newer (Python 3.11 recommended)
- `pip`

## 2. Create and activate a virtual environment

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install dependencies

From the project directory:

```bash
pip install -r requirements.txt
```

## 4. Train the models

Run:

```bash
python train.py
```

The script:
1. Creates 5,000 synthetic 7–10-day histories.
2. Trains a Random Forest classifier and Random Forest regressor.
3. Prints a held-out accuracy report and temperature mean absolute error.
4. Saves both models under `models/` using `joblib`.

Training can take a short while depending on your computer. Re-run this command whenever you change the training logic. **Only load model artifacts you trust**, since pickle/joblib-based files can execute code when loaded.

## 5. Start the Flask server

```bash
python app.py
```

The API listens at `http://127.0.0.1:5000`. The built-in development server is for local testing only; use a production WSGI server and appropriate security configuration for deployment.

Useful endpoints:
- `GET /` — API overview and example request shape
- `GET /health` — basic health check
- `POST /predict` — prediction endpoint

## 6. Send a prediction request

The `history` list must contain **7 to 10 daily records**, in chronological order from oldest to newest. Each record requires:
- `temperature`: numeric degrees Celsius
- `humidity`: relative humidity from 0 to 100
- `wind_speed`: numeric wind speed (the training examples use a generic speed scale; use one consistent unit in your real dataset)
- `condition`: `Sunny`, `Rainy`, or `Cloudy`

### Example with curl

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "history": [
      {"temperature": 26.1, "humidity": 68, "wind_speed": 9.2, "condition": "Cloudy"},
      {"temperature": 27.0, "humidity": 65, "wind_speed": 8.0, "condition": "Sunny"},
      {"temperature": 25.8, "humidity": 77, "wind_speed": 12.1, "condition": "Cloudy"},
      {"temperature": 24.9, "humidity": 84, "wind_speed": 16.0, "condition": "Rainy"},
      {"temperature": 25.4, "humidity": 81, "wind_speed": 13.2, "condition": "Rainy"},
      {"temperature": 26.7, "humidity": 73, "wind_speed": 10.1, "condition": "Cloudy"},
      {"temperature": 27.2, "humidity": 66, "wind_speed": 7.4, "condition": "Sunny"}
    ]
  }'
```

### Example response

Actual values depend on the model artifacts generated on your machine.

```json
{
  "predicted_condition": "Cloudy",
  "predicted_temperature_c": 26.48,
  "class_probabilities": {
    "Cloudy": 0.51,
    "Rainy": 0.12,
    "Sunny": 0.37
  },
  "confidence": 0.51,
  "input_days": 7,
  "note": "Demo model trained on synthetic data; predictions are illustrative, not a forecast."
}
```

The example probabilities above are illustrative. `confidence` is the largest class probability, not a guarantee that the prediction is correct.

## 7. Test the API

In a second terminal, with the server running:

```bash
curl http://127.0.0.1:5000/health
```

You should receive:

```json
{"status":"ok"}
```

For a request body with fewer than 7 or more than 10 days, missing fields, invalid values, or an unsupported condition, the API returns a JSON error with HTTP status `400`. If the model files do not exist yet, `/predict` returns HTTP `503`; run `python train.py`.

## 8. Python request example

```python
import requests

payload = {
    "history": [
        {"temperature": 26, "humidity": 70, "wind_speed": 8, "condition": "Cloudy"}
    ] * 7
}
response = requests.post("http://127.0.0.1:5000/predict", json=payload, timeout=10)
response.raise_for_status()
print(response.json())
```

This example repeats one record only to demonstrate request syntax. Real input should contain actual measurements for distinct consecutive days.

## 9. Improving model quality

For meaningful operational forecasts:
1. Replace synthetic generation in `train.py` with real historical observations for the target location.
2. Ensure feature units and condition labels are consistent.
3. Use chronological train/validation/test splits to avoid temporal leakage.
4. Compare against simple baselines and report class-wise precision/recall, temperature MAE/RMSE, and calibration.
5. Retrain periodically and monitor performance drift.
6. Consider additional predictors such as pressure, precipitation, cloud cover, seasonality, and official forecast data.

## License

Provided as an educational starter project. A GNU Public license-v3.o has been added .


## Deployment

THE project has been deployed in render.
