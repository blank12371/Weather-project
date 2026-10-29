"""Generate synthetic weather sequences and train/save classifier and regressor."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

from model import MODEL_DIR, CONDITION_TO_CODE

RNG = np.random.default_rng(42)
N_SAMPLES = 5000
CONDITIONS = np.array(["Sunny", "Rainy", "Cloudy"])
FEATURE_COLUMNS = None


def make_sample():
    """Create a synthetic 7-10 day window and a plausible next-day target."""
    n_days = int(RNG.integers(7, 11))
    season_temp = RNG.uniform(5, 32)
    humidity_base = RNG.uniform(35, 92)
    wind_base = RNG.uniform(1, 28)
    rows = []
    previous_condition = str(RNG.choice(CONDITIONS))
    for _ in range(n_days):
        temp = season_temp + RNG.normal(0, 3.2)
        humidity = np.clip(humidity_base + RNG.normal(0, 10), 15, 100)
        wind = np.clip(wind_base + RNG.normal(0, 5), 0, 65)
        # Synthetic rule to generate conditions with some overlap/noise.
        rain_score = (humidity - 65) / 20 + (wind - 15) / 35 - (temp - 18) / 30
        if RNG.random() < 0.12:
            condition = str(RNG.choice(CONDITIONS))
        elif rain_score > 0.8:
            condition = "Rainy"
        elif humidity > 72 or RNG.random() < 0.25:
            condition = "Cloudy"
        else:
            condition = "Sunny"
        rows.append({
            "temperature": float(temp),
            "humidity": float(humidity),
            "wind_speed": float(wind),
            "condition": condition,
        })
        previous_condition = condition

    temps = np.array([r["temperature"] for r in rows])
    hums = np.array([r["humidity"] for r in rows])
    winds = np.array([r["wind_speed"] for r in rows])
    codes = np.array([CONDITION_TO_CODE[r["condition"]] for r in rows])
    f = {
        "days_count": len(rows),
        "temp_mean": temps.mean(),
        "temp_min": temps.min(),
        "temp_max": temps.max(),
        "temp_last": temps[-1],
        "temp_change": temps[-1] - temps[0],
        "humidity_mean": hums.mean(),
        "humidity_last": hums[-1],
        "wind_mean": winds.mean(),
        "wind_last": winds[-1],
        "sunny_fraction": np.mean(codes == CONDITION_TO_CODE["Sunny"]),
        "rainy_fraction": np.mean(codes == CONDITION_TO_CODE["Rainy"]),
        "cloudy_fraction": np.mean(codes == CONDITION_TO_CODE["Cloudy"]),
        "condition_last_code": codes[-1],
    }
    padded = [rows[0]] * (10 - len(rows)) + rows
    for lag, row in enumerate(reversed(padded[-10:]), start=1):
        f[f"lag{lag}_temperature"] = row["temperature"]
        f[f"lag{lag}_humidity"] = row["humidity"]
        f[f"lag{lag}_wind_speed"] = row["wind_speed"]
        f[f"lag{lag}_condition_code"] = CONDITION_TO_CODE[row["condition"]]

    # Next-day targets depend on recent conditions and trend, with random variation.
    rain_signal = (hums.mean() - 62) / 18 + (winds[-1] - 12) / 28 - (temps[-1] - 18) / 25
    if RNG.random() < 0.10:
        next_condition = str(RNG.choice(CONDITIONS))
    elif rain_signal > 0.85:
        next_condition = "Rainy"
    elif hums[-1] > 70 or rows[-1]["condition"] == "Cloudy":
        next_condition = "Cloudy"
    else:
        next_condition = "Sunny"
    next_temp = 0.55 * temps[-1] + 0.30 * temps.mean() + 0.15 * temps[0]
    next_temp += (temps[-1] - temps[0]) * 0.08 + RNG.normal(0, 2.0)
    return f, next_condition, float(next_temp)


def main():
    samples = [make_sample() for _ in range(N_SAMPLES)]
    X = pd.DataFrame([s[0] for s in samples])
    y_class = np.array([s[1] for s in samples])
    y_temp = np.array([s[2] for s in samples])
    X_train, X_test, yc_train, yc_test, yt_train, yt_test = train_test_split(
        X, y_class, y_temp, test_size=0.2, random_state=42, stratify=y_class
    )
    classifier = RandomForestClassifier(
        n_estimators=220, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    regressor = RandomForestRegressor(
        n_estimators=220, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    classifier.fit(X_train, yc_train)
    regressor.fit(X_train, yt_train)
    predicted_class = classifier.predict(X_test)
    predicted_temp = regressor.predict(X_test)
    print(f"Classifier accuracy: {accuracy_score(yc_test, predicted_class):.3f}")
    print("Classification report:")
    print(classification_report(yc_test, predicted_class, zero_division=0))
    print(f"Temperature MAE: {mean_absolute_error(yt_test, predicted_temp):.2f} °C")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_DIR / "weather_classifier.joblib")
    joblib.dump(regressor, MODEL_DIR / "temperature_regressor.joblib")
    print(f"Saved model artifacts to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
