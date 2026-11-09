"""
Train the house-price model and save it with joblib.

Run once before starting the app:   python train_model.py
Creates: model.joblib  (model + metadata the app needs)
"""
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Friendly names + units for the app's sliders
LABELS = {
    "MedInc": "Median income (x $10,000)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms per household",
    "AveBedrms": "Average bedrooms per household",
    "Population": "Block population",
    "AveOccup": "Average occupants per household",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}


def load_data():
    try:
        d = fetch_california_housing()
        return d.data, d.target, list(d.feature_names), "California Housing"
    except Exception as e:  # offline fallback so the pipeline can still be tested
        print(f"WARNING: could not download dataset ({type(e).__name__}). "
              "Using SYNTHETIC data - connect to the internet for the real model.")
        rng = np.random.default_rng(RANDOM_STATE)
        n = 5000
        X = np.column_stack([
            rng.lognormal(1.2, 0.5, n).clip(0.5, 15),   # MedInc
            rng.integers(1, 52, n),                      # HouseAge
            rng.normal(5.4, 1.2, n).clip(2, 10),         # AveRooms
            rng.normal(1.1, 0.1, n).clip(0.8, 2),        # AveBedrms
            rng.lognormal(7.1, 0.7, n).clip(5, 20000),   # Population
            rng.normal(3.0, 0.6, n).clip(1, 8),          # AveOccup
            rng.uniform(32.5, 42, n),                    # Latitude
            rng.uniform(-124.3, -114.3, n),              # Longitude
        ])
        y = (0.45 * X[:, 0] + 0.005 * X[:, 1] + 0.05 * X[:, 2]
             + rng.normal(0, 0.4, n) + 0.4).clip(0.15, 5)
        return X, y, list(LABELS.keys()), "Synthetic (offline fallback)"


def main():
    X, y, names, source = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    baseline = LinearRegression().fit(X_tr, y_tr)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                          random_state=RANDOM_STATE).fit(X_tr, y_tr)

    pred = model.predict(X_te)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "mse": float(mean_squared_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
        "baseline_r2": float(r2_score(y_te, baseline.predict(X_te))),
    }
    imp = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=RANDOM_STATE)

    # Slider ranges: 1st-99th percentile avoids absurd outliers
    features = {}
    for i, n in enumerate(names):
        lo, hi = np.percentile(X[:, i], [1, 99])
        features[n] = {"label": LABELS.get(n, n), "min": float(lo), "max": float(hi),
                       "default": float(np.median(X[:, i]))}

    joblib.dump({
        "model": model,
        "features": features,
        "feature_order": names,
        "metrics": metrics,
        "importance": dict(zip(names, imp.importances_mean.tolist())),
        "median_price": float(np.median(y)),
        "source": source,
    }, "model.joblib", compress=3)

    print(f"Data: {source}")
    print(f"Test RMSE: {metrics['rmse']:.3f} | R2: {metrics['r2']:.3f} "
          f"(linear baseline R2: {metrics['baseline_r2']:.3f})")
    print("Saved model.joblib")


if __name__ == "__main__":
    main()
