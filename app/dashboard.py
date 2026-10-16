from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json

class Dashboard:
    def __init__(self):
        self.predictor = Predictor()

    # ---------- CONFIG ----------
    st.set_page_config(
        page_title="Solar Power Prediction",
        page_icon="⚡",
        layout="wide",
    )

    ROOT = Path(__file__).resolve().parent.parent
    RAW_DATA = ROOT / "data" / "raw" / "data_set_raw.csv"
    PROCESSED_DIR = ROOT / "data" / "processed"
    MODELS_DIR = ROOT / "models"

    # ---------- DATA LOADING ----------
    @st.cache_data
    def load_raw_data() -> pd.DataFrame:
        """Load the raw dataset."""
        return pd.read_csv(RAW_DATA)

    # ---------- SIDEBAR NAVIGATION ----------
    def page_initial(self):

        st.sidebar.title("Solar Power Prediction")
        st.sidebar.markdown("---")

        page = st.sidebar.radio(
            "Navigate to:",
            [
                "📊 Data Overview",
                "🔍 Interactive Exploration",
                "🔮 Prediction",
                "📈 Model Performance",
            ],
        )

        st.sidebar.markdown("---")
        st.sidebar.caption("CST8504 — Assignment 1")

        return page
    # ------------------------------------------

    # ============================================================
    # PAGE 1: DATA OVERVIEW
    # ============================================================
    def page_data_overview():
        st.title("📊 Data Overview")
        st.markdown(
            "The NASA POWER dataset contains 368 weather features for any geographic coordinate."
            "Here is a subset of it with 7 features that have the greatest impact on solar power generation."
            "The range of dates is from 2025-07-01 to 2026-06-30 and the City is Tamil Nadu (India)."
            "Panel specifications: - 3 photovoltaic solar panels (total installed capacity: 3kW)."
            "                      - panel area approximatly 15m²"
            "The goal is to predict the generated energy on this specific system."
        )

        df = load_raw_data()
    
        # Top-level metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Instances", f"{df.shape[0]:,}")
        col2.metric("Features", df.shape[1] - 1)
   #     col3.metric("Edible (e)", f"{(df['class'] == 'e').sum():,}")
   #     col4.metric("Poisonous (p)", f"{(df['class'] == 'p').sum():,}")

        st.markdown("---")

        # Preview table
        st.subheader("📋 Dataset Preview")
        n_rows = st.slider("Rows to preview", 5, 50, 10)
        st.dataframe(df.head(n_rows), use_container_width=True)

        st.markdown("---")

        # Summary statistics
        st.subheader("📈 Summary Statistics")
        summary = pd.DataFrame({
            "Unique Values": df.nunique(),
            "Missing (?)": (df == "?").sum(),
            "Most Common": [df[col].mode()[0] for col in df.columns],
        })
        st.dataframe(summary, use_container_width=True)

        st.markdown("---")

        # Class distribution
        st.subheader("⚖️ Target Class Balance")
        class_counts = df["class"].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]
        class_counts["Class"] = class_counts["Class"].map({"e": "Edible", "p": "Poisonous"})
        fig = px.bar(
            class_counts, x="Class", y="Count", color="Class",
            title="Distribution of Edible vs Poisonous Mushrooms",
            color_discrete_map={"Edible": "#2ecc71", "Poisonous": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "✅ The dataset is **fairly balanced** (~52% edible / ~48% poisonous), "
            "so accuracy is a reasonable metric."
        )

    # -------------------------------------------------
    st.title("Solar Power Production Prediction")

    st.write("Features for prediction:")

    ghi = st.number_input("GHI - Globar Horizontal Irradiance (W/m²)") 
    ghi_clear = st.number_input("GHI Clear Sky - (W/m²)") 
    temp = st.number_input("Temperature (C)")
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    wind = st.number_input("Wind (m/s)") 
    cloud = st.slider("Cloud (%)", 0, 100, 50)
    rain = st.number_input("Precipitation Corrected (mm/day)") 

    # Load Predictor
    predictor = Predictor()

    input_df = pd.DataFrame([{
        "ghi": ghi,
        "ghi_clear":gui_clear,
        "temp": temp,
        "humidity": humidity,
        "wind":wind,
        "clouds":clouds,
        "rain":rain,
        "actual":actual,
        "predicted":predicted
    }])

    pred = predictor.predict(input_df)
    st.success(f"Previsão: {pred:.2f} kWh")

    # Plot prediction
    fig = px.line(
        x=[0, 1],
        y=[0, pred],
        markers=True,
        title="Energy predicted"
    )

    fig.add_scatter(
        x=[1],
        y=[pred],
        mode="markers",
        marker=dict(size=12, color="red"),
        name="Predicted value"
    )

    st.plotly_chart(fig, use_container_width=True)
    #----------------

    def run(self):
        st.set_page_config(page_title="Solar Dashboard", layout="wide")
        st.title("🔆 Solar Energy Dashboard")

        page = self.page_initial()

        if page == "📊 Data Overview":
            self.page_overview()

        # elif page == "🔍 Interactive Exploration":
        #     self.page_exploration()

       # elif page == "🔮 Prediction":
       #     self.page_prediction()

       # elif page == "📈 Model Performance":
       #     self.page_performance()