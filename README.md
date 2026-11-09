# 🏠 House Price Predictor (Streamlit)

An end-to-end ML project: train a scikit-learn model, save it with `joblib`,
and serve it through an interactive Streamlit app with live predictions.

## Features
- 8 input sliders (ranges auto-derived from the training data)
- Real-time price prediction with an error range (±1 RMSE)
- Comparison against the median house price
- Map of the chosen location, feature-importance chart, model info tab

## Run locally
```bash
pip install -r requirements.txt
python train_model.py      # trains the model, creates model.joblib (needs internet once)
streamlit run app.py       # opens http://localhost:8501
```

## Deploy free on Streamlit Community Cloud
1. Run `python train_model.py` and confirm `model.joblib` exists (it is small, ~1 MB).
2. Push this folder to a **public GitHub repo** (include `model.joblib`; remove it from `.gitignore` if present).
3. Go to https://share.streamlit.io → **New app** → pick the repo, branch, and `app.py`.
4. Click **Deploy**. You get a public URL to put on your CV/portfolio.

## Project structure
| File | Purpose |
|---|---|
| `train_model.py` | Loads data, train/test split, trains model, evaluates, saves `model.joblib` |
| `app.py` | Streamlit UI: loads model, sliders, predictions, charts |
| `requirements.txt` | Dependencies for local + cloud deploy |

## Model
HistGradientBoostingRegressor, compared to a Linear Regression baseline.
Metrics (RMSE, MSE, R²) are computed on a held-out 20% test set and shown in the app.

## Ideas to extend
- Add a model-comparison dropdown (Linear / Random Forest / Gradient Boosting)
- Add SHAP explanations for each individual prediction
- Let users upload a CSV for batch predictions
