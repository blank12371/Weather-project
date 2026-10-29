"""Feature preparation and prediction helpers for the weather models."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"
CLASSIFIER_PATH = MODEL_DIR / "weather_classifier.joblib"
REGRESSOR_PATH = MODEL_DIR / "temperature_regressor.joblib"
CONDITIONS = ("Sunny", "Rainy", "Cloudy")
CONDITION_TO_CODE = {"Sunny": 0, "Cloudy": 1, "Rainy": 2}
REQUIRED_FIELDS = ("temperature", "humidity", "wind_speed", "condition")


class ModelNotTrainedError(RuntimeError):
    """Raised when trained model artifacts are not available."""


def _number(value, field, day_index):
    if isinstance(value, bool):
        raise ValueError(f"Day {day_index}: '{field}' must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Day {day_index}: '{field}' must be a number.") from None
    if not np.isfinite(number):
        raise ValueError(f"Day {day_index}: '{field}' must be finite.")
    return number


def validate_history(history):
    if not 7 <= len(history) <= 10:
        raise ValueError("'history' must contain between 7 and 10 daily records.")
    rows = []
    for index, day in enumerate(history, start=1):
        if not isinstance(day, dict):
            raise ValueError(f"Day {index} must be a JSON object.")
        missing = [field for field in REQUIRED_FIELDS if field not in day]
        if missing:
            raise ValueError(f"Day {index} is missing fields: {', '.join(missing)}.")
        temperature = _number(day["temperature"], "temperature", index)
        humidity = _number(day["humidity"], "humidity", index)
        wind_speed = _number(day["wind_speed"], "wind_speed", index)
        condition = str(day["condition"]).strip().title()
        if condition not in CONDITIONS:
            raise ValueError(
                f"Day {index}: condition must be one of Sunny, Rainy, or Cloudy."
            )
        if not -90 <= temperature <= 65:
            raise ValueError(f"Day {index}: temperature must be between -90 and 65 °C.")
        if not 0 <= humidity <= 100:
            raise ValueError(f"Day {index}: humidity must be between 0 and 100.")
        if wind_speed < 0 or wind_speed > 400:
            raise ValueError(f"Day {index}: wind_speed must be between 0 and 400.")
        rows.append({
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "condition": condition,
        })
    return rows


def _features(rows):
    """Convert an ordered history window into the same features used for training."""
    temperatures = np.array([r["temperature"] for r in rows], dtype=float)
    humidities = np.array([r["humidity"] for r in rows], dtype=float)
    winds = np.array([r["wind_speed"] for r in rows], dtype=float)
    condition_codes = np.array([CONDITION_TO_CODE[r["condition"]] for r in rows], dtype=float)
    features = {
        "days_count": len(rows),
        "temp_mean": temperatures.mean(),
        "temp_min": temperatures.min(),
        "temp_max": temperatures.max(),
        "temp_last": temperatures[-1],
        "temp_change": temperatures[-1] - temperatures[0],
        "humidity_mean": humidities.mean(),
        "humidity_last": humidities[-1],
        "wind_mean": winds.mean(),
        "wind_last": winds[-1],
        "sunny_fraction": np.mean(condition_codes == CONDITION_TO_CODE["Sunny"]),
        "rainy_fraction": np.mean(condition_codes == CONDITION_TO_CODE["Rainy"]),
        "cloudy_fraction": np.mean(condition_codes == CONDITION_TO_CODE["Cloudy"]),
        "condition_last_code": condition_codes[-1],
    }
    # Include lagged values; short windows are padded by repeating the oldest available
    # value, making the feature vector fixed-width for 7-, 8-, 9-, and 10-day inputs.
    padded = [rows[0]] * (10 - len(rows)) + rows
    for lag, row in enumerate(reversed(padded[-10:]), start=1):
        features[f"lag{lag}_temperature"] = row["temperature"]
        features[f"lag{lag}_humidity"] = row["humidity"]
        features[f"lag{lag}_wind_speed"] = row["wind_speed"]
        features[f"lag{lag}_condition_code"] = CONDITION_TO_CODE[row["condition"]]
    return pd.DataFrame([features])


def predict_weather(history):
    if not CLASSIFIER_PATH.exists() or not REGRESSOR_PATH.exists():
        raise ModelNotTrainedError("Trained model files were not found.")
    rows = validate_history(history)
    x = _features(rows)
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
    category = str(classifier.predict(x)[0])
    probabilities = classifier.predict_proba(x)[0]
    confidence = {str(label): round(float(prob), 4)
                  for label, prob in zip(classifier.classes_, probabilities)}
    temperature = float(regressor.predict(x)[0])
    return {
        "predicted_condition": category,
        "predicted_temperature_c": round(temperature, 2),
        "class_probabilities": confidence,
        "confidence": round(float(max(probabilities)), 4),
        "input_days": len(rows),
        "note": "Demo model trained on synthetic data; predictions are illustrative, not a forecast."
    }
