"""
Streamlit app: California house price predictor.
Run locally:  streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource  # load the model once, not on every slider move
def load_bundle():
    return joblib.load("model.joblib")


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("`model.joblib` not found. Run `python train_model.py` first.")
    st.stop()

model = bundle["model"]
features = bundle["features"]
order = bundle["feature_order"]
m = bundle["metrics"]

st.title("🏠 House Price Predictor")
st.caption("A gradient-boosting model trained on the California Housing dataset. "
           "Move the sliders and the prediction updates instantly.")

# ---------------- Sidebar: input sliders ----------------
st.sidebar.header("Property details")
if st.sidebar.button("Reset to typical values"):
    for k in order:
        st.session_state.pop(k, None)

inputs = {}
for name in order:
    f = features[name]
    lo, hi, default = f["min"], f["max"], f["default"]
    step = 1.0 if hi - lo > 50 else 0.01
    inputs[name] = st.sidebar.slider(
        f["label"], min_value=float(lo), max_value=float(hi),
        value=float(default), step=step, key=name)

# ---------------- Real-time prediction ----------------
X = pd.DataFrame([inputs])[order]
price = float(model.predict(X.values)[0]) * 100_000
median = bundle["median_price"] * 100_000
delta = price - median

col1, col2, col3 = st.columns(3)
col1.metric("Predicted price", f"${price:,.0f}",
            f"{delta:+,.0f} vs. median (${median:,.0f})")
col2.metric("Typical error (RMSE)", f"±${m['rmse'] * 100_000:,.0f}")
col3.metric("Model R² (test set)", f"{m['r2']:.2f}",
            f"{m['r2'] - m['baseline_r2']:+.2f} vs. linear regression")

low, high = max(price - m["rmse"] * 100_000, 0), price + m["rmse"] * 100_000
st.info(f"Likely range (±1 RMSE): **${low:,.0f} – ${high:,.0f}**")

# ---------------- Tabs ----------------
tab_map, tab_imp, tab_about = st.tabs(["📍 Location", "📊 What drives the price", "ℹ️ About"])

with tab_map:
    st.map(pd.DataFrame({"lat": [inputs["Latitude"]], "lon": [inputs["Longitude"]]}),
           zoom=5)

with tab_imp:
    imp = pd.Series(bundle["importance"]).rename(index=lambda n: features[n]["label"])
    st.bar_chart(imp.sort_values())
    st.caption("Permutation importance: how much the model's accuracy drops when a "
               "feature is shuffled. Bigger = more influential.")

with tab_about:
    st.markdown(f"""
**Data:** {bundle['source']}  
**Model:** HistGradientBoostingRegressor (scikit-learn)  
**Test RMSE:** {m['rmse']:.3f} (≈ ${m['rmse'] * 100_000:,.0f}) · **MSE:** {m['mse']:.3f} · **R²:** {m['r2']:.3f}

**Limitations:** prices are 1990 census block-group medians, so they do not reflect
today's market. This is a learning project, not financial advice.
""")
    with st.expander("Show model input"):
        st.dataframe(X)
