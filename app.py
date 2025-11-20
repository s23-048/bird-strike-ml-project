# app.py
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "model/bird_strike_pipeline.pkl"
DATA_PATH = "data/faa_bird_strike_sample_10k.csv"

SEVERITY_LABELS = {
    0: "No / None",
    1: "Minor",
    2: "Severe",
}


@st.cache_data
def load_dataset(path: str):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.upper() for c in df.columns]
        return df
    return None


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def main():
    st.title("✈️ Bird Strike Damage Prediction")
    st.write(
        """
This app uses a machine learning model trained on bird strike incident data
to predict the **severity of aircraft damage** based on flight conditions.
"""
    )

    pipeline = load_model(MODEL_PATH)
    if pipeline is None:
        st.error("Model file not found. Please run `python train.py` first to train and save the model.")
        return

    df = load_dataset(DATA_PATH)

    st.sidebar.header("Input Features")

    if df is not None:
        aircraft_options = sorted(df["AIRCRAFT"].astype(str).unique().tolist())
        phase_options = sorted(df["PHASE_OF_FLIGHT"].astype(str).unique().tolist())
        species_options = sorted(df["SPECIES"].astype(str).unique().tolist())
        time_options = sorted(df["TIME_OF_DAY"].astype(str).unique().tolist())
    else:
        aircraft_options = ["NARROWBODY", "WIDEBODY", "BUSINESS JET", "GA SINGLE"]
        phase_options = ["TAKEOFF", "LANDING", "CLIMB", "APPROACH", "CRUISE"]
        species_options = ["GULL", "WATERFOWL", "RAPTOR", "SONGBIRD"]
        time_options = ["DAY", "NIGHT", "DAWN", "DUSK"]

    aircraft = st.sidebar.selectbox("Aircraft type", aircraft_options)
    phase = st.sidebar.selectbox("Phase of flight", phase_options)
    species = st.sidebar.selectbox("Bird species", species_options)
    time_of_day = st.sidebar.selectbox("Time of day", time_options)
    altitude = st.sidebar.number_input(
        "Altitude (feet AGL)",
        min_value=-1,
        max_value=50000,
        value=500,
        step=100,
        help="Approximate height above ground level when strike occurred.",
    )

    st.subheader("Enter Conditions and Predict Damage Severity")

    if st.button("Predict damage severity"):
        input_df = pd.DataFrame(
            [
                {
                    "AIRCRAFT": aircraft,
                    "PHASE_OF_FLIGHT": phase,
                    "SPECIES": species,
                    "TIME_OF_DAY": time_of_day,
                    "HEIGHT": altitude,
                }
            ]
        )

        pred = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0]

        severity_text = SEVERITY_LABELS.get(int(pred), f"Class {pred}")

        st.markdown(f"### 🔮 Predicted severity: **{severity_text}**")

        prob_df = pd.DataFrame(
            {
                "Severity class": [SEVERITY_LABELS.get(i, f"Class {i}") for i in range(len(proba))],
                "Probability": proba,
            }
        )

        st.write("Class probabilities:")
        st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

    st.markdown("---")
    st.caption("MLT Course Project • Bird Strike Damage Prediction using Random Forest and FAA-style data.")


if __name__ == "__main__":
    main()
