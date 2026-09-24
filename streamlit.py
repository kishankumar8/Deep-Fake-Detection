import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="centered",
)

st.title("🕵️ Deepfake Media Detector")
st.write(
    "Enter the details of a piece of media below and the model will estimate "
    "whether it's **Real** or **Fake**, based on metadata and forensic scores."
)

MODEL_PATH = "deepfake_model.pkl"

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_model(MODEL_PATH)

if model is None:
    st.error(
        f"Couldn't find `{MODEL_PATH}` in the app folder. "
        "Make sure `deepfake_model.pkl` (produced by your notebook's "
        "`joblib.dump(model, \"deepfake_model.pkl\")` step) is placed next to `app.py`."
    )
    st.stop()

# The trained columns (after pandas get_dummies) — sklearn stores this
# automatically when a model is fit on a DataFrame.
try:
    trained_columns = list(model.feature_names_in_)
except AttributeError:
    st.error(
        "This model doesn't expose `feature_names_in_` (it may have been trained "
        "on a plain numpy array instead of a DataFrame). Re-save the model after "
        "fitting it directly on the pandas DataFrame `x_train`."
    )
    st.stop()

# Known base columns from the notebook
NUMERIC_BASE = [
    "face_count",
    "lip_sync_score",
    "visual_artifacts_score",
    "compression_level",
    "lighting_inconsistency_score",
]
CATEGORICAL_BASE = [
    "media_type",
    "content_category",
    "audio_present",
    "source_platform",
    "generation_method",
]

# Figure out which numeric columns actually exist in the trained model
numeric_cols = [c for c in NUMERIC_BASE if c in trained_columns]

# Figure out categorical options by parsing the dummy column names,
# e.g. "media_type_Video" -> base="media_type", option="Video"
categorical_options = {}
for base in CATEGORICAL_BASE:
    prefix = base + "_"
    opts = [c[len(prefix):] for c in trained_columns if c.startswith(prefix)]
    if opts:
        categorical_options[base] = sorted(opts)

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
with st.form("detection_form"):
    st.subheader("Forensic scores")
    st.caption("0 = no anomaly detected, 1 = strong anomaly detected")

    numeric_values = {}
    col1, col2 = st.columns(2)
    slider_cols = [c for c in numeric_cols if c != "face_count"]
    for i, col in enumerate(slider_cols):
        target = col1 if i % 2 == 0 else col2
        label = col.replace("_", " ").title()
        numeric_values[col] = target.slider(label, 0.0, 1.0, 0.5, 0.01)

    if "face_count" in numeric_cols:
        numeric_values["face_count"] = st.number_input(
            "Number of faces detected", min_value=0, max_value=20, value=1, step=1
        )

    st.subheader("Media details")
    categorical_values = {}
    for base, opts in categorical_options.items():
        label = base.replace("_", " ").title()
        choice = st.selectbox(label, options=["(unspecified / other)"] + opts)
        categorical_values[base] = None if choice == "(unspecified / other)" else choice

    submitted = st.form_submit_button("🔍 Analyze")

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if submitted:
    row = {col: 0 for col in trained_columns}

    # media_id (or similar leftover id column) has no real predictive meaning —
    # default it to 0 if the trained model still expects it.
    for leftover in set(trained_columns) - set(numeric_cols) - {
        c for base, opts in categorical_options.items() for c in trained_columns if c.startswith(base + "_")
    }:
        row[leftover] = 0

    for col, val in numeric_values.items():
        row[col] = val

    for base, val in categorical_values.items():
        if val is not None:
            dummy_col = f"{base}_{val}"
            if dummy_col in row:
                row[dummy_col] = 1

    input_df = pd.DataFrame([row], columns=trained_columns)

    prediction = model.predict(input_df)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

    st.divider()
    if str(prediction).lower() == "fake":
        st.error(f"### 🚨 Prediction: **{prediction}**")
    else:
        st.success(f"### ✅ Prediction: **{prediction}**")

    if proba is not None:
        st.write("Confidence breakdown:")
        prob_df = pd.DataFrame({"Class": classes, "Probability": proba}).set_index("Class")
        st.bar_chart(prob_df)
        confidence = max(proba) * 100
        st.caption(f"Model confidence: {confidence:.1f}%")

    with st.expander("See raw input sent to the model"):
        st.dataframe(input_df)

st.divider()
st.caption(
    "⚠️ This tool relies on metadata and precomputed forensic scores rather than "
    "analyzing raw video/audio itself, and its output is a probabilistic estimate — "
    "not proof of authenticity."
)