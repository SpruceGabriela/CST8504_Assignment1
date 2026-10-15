import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# ------------------------------------------------------------
# Ensure Python can find the src/ package
# ------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predictor import Predictor


# ============================================================
# MAIN DASHBOARD CLASS (OOP)
# ============================================================
class Dashboard:
    def __init__(self):
        """Load prepared datasets and trained model."""

        # Load processed datasets
        self.raw = pd.read_csv("data/raw/data_set_raw.csv")
        self.raw2 = pd.read_csv("data/raw/daily_generation.csv")
        self.prepared = pd.read_csv("data/processed/data_set_prepared.csv")
        self.train = pd.read_csv("data/processed/train.csv")
        self.validation = pd.read_csv("data/processed/validation.csv")
        self.test = pd.read_csv("data/processed/test.csv")

        # Load model + scaler
        self.predictor = Predictor(
            model_path="models/linear_regression.pkl", scaler_path="models/scaler.pkl"
        )

    # ============================================================
    # PAGE 1 — DATA OVERVIEW
    # ============================================================
    def page_data_overview(self):
        st.header("📊 Data Overview")

        st.subheader("Raw Nasa Power Data Preview (meteorological data)")
        col1, col2, col3 = st.columns([1, 1, 5])  # both columns small
        with col1:
            start = st.number_input(
                "Start row", min_value=0, max_value=len(self.raw) - 1, value=0
            )

        with col2:
            end = st.number_input(
                "End row",
                min_value=1,
                max_value=len(self.raw),
                value=10,
            )
        st.dataframe(self.raw.iloc[start:end])

        # --------------------------
        #  Graph Feature Overtime
        # --------------------------
        st.markdown("""
        ### Feature Overtime  
        Choose a feature below to visualize its behavior over time.
        """)

        # Use RAW dataset
        df = self.raw.copy()

        # Convert date column if it exists
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

        # Create a mapping between raw column names and readable labels
        feature_labels = {
            "ALLSKY_SFC_SW_DWN": "GHI - Globar Horizontal Irradiance (W/m²)",
            "CLRSKY_SFC_SW_DWN": "GHI Clear Sky - (W/m²)",
            "T2M": "Temperature (C)",
            "RH2M": "Humidity (%)",
            "WS10M": "Wind (m/s)",
            "CLOUD_AMT": "Cloud (%)",
            "PRECTOTCORR": "Precipitation Corrected (mm/day)",
            "date": "Date"
        }

        # Remove "date" from labels list
        labels_without_date = {
            key: value for key, value in feature_labels.items() if key != "date"
        }
        # Build dropdown with readable labels
        col1, col2 = st.columns([1, 3])  # col1 é estreita, col2 é larga

        with col1:
            feature_label = st.selectbox(
                "Select a feature:",
                list(labels_without_date.values())
            )

        # Find the corresponding raw column name
        feature = [k for k, v in feature_labels.items() if v == feature_label][0]

        #st.subheader(f"{feature_label} Over Time")

        # Line chart
        if "date" in df.columns:
            fig = px.line(
                df,
                x="date",
                y=feature,
                title=f"{feature_label} Over Time",
                markers=True,
                labels={
                    "date": "Date",          # 🔥 renomeia eixo X
                    feature: feature_label   # 🔥 renomeia eixo Y
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            fig = px.line(
                df,
                y=feature,
                title=f"{feature_label} Line Chart",
                markers=True,
                labels={
                    "index": "Index",        # eixo X quando não há date
                    feature: feature_label   # eixo Y
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------

        st.subheader("Raw Daily Generation Data Preview (Generation data)")
        col11, col12, col13 = st.columns([1, 1, 5])  # both columns small
        with col11:
            start1 = st.number_input(
                "Start row", min_value=0, max_value=len(self.raw2) - 1, value=0
            )

        with col12:
            end1 = st.number_input(
                "End row",
                min_value=1,
                max_value=len(self.raw2),
                value=10,
            )
        st.dataframe(self.raw2.iloc[start1:end1])

        st.subheader("Training Data Preview (Prepared Data)")
        col111, col112, col113 = st.columns([1, 1, 5])  # both columns small
        with col111:
            start2 = st.number_input(
                "Start row", min_value=0, max_value=len(self.train) - 1, value=0
            )

        with col112:
            end2 = st.number_input(
                "End row",
                min_value=1,
                max_value=len(self.train),
                value=10,
            )
        st.dataframe(self.train.iloc[start2:end2])

        st.subheader("📏 Dataset Sizes after split (70/15/15)")

        st.markdown(f"""
        - 🟢 **Train Dataset:** {self.train.shape[0]} instances  
        - 🟡 **Validation Dataset:** {self.validation.shape[0]} instances  
        - 🔵 **Test Dataset:** {self.test.shape[0]} instances  
        """)
     

    # ============================================================
    # MAIN NAVIGATION
    # ============================================================
    def run(self):
        st.set_page_config(layout="wide") 

        st.title("🔆 Solar Energy Production Dashboard")

        # 🔧 CSS for navbar on top of the page
        st.markdown("""
        <style>
        /* Navbar fixa */
        div[data-testid="stRadio"] {
            position: fixed;
            top: 10px;
            z-index: 999;
            background-color: #0e1117;
            padding: 15px 0px 20px 0px;
            width: 100%;
            border-bottom: 1px solid #333;
        }

        /* Push down the content */
        .main {
            margin-top: 70px !important;
        }

        /* Remove lateral spaces */
        .block-container {
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # 🔘 Top Menu
        menu = st.radio(
            "",
            ["📊 Data Overview", "⚡ Prediction", "📉 Model Performance"],
            horizontal=True
        )

        st.markdown("---")

        # 🔍 Conteúdo em largura total
        if menu == "📊 Data Overview":
            self.page_data_overview()
        elif menu == "⚡ Prediction":
            self.page_prediction()
        elif menu == "📉 Model Performance":
            self.page_performance()


# ============================================================
# RUN DASHBOARD
# ============================================================
if __name__ == "__main__":
    Dashboard().run()
