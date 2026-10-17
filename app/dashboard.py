import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import plotly.graph_objects as go

# ------------------------------------------------------------
# Ensure Python can find the src/ package
# ------------------------------------------------------------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predictor import Predictor
from src.data_processor import DataProcessor


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

        # Create a mapping between raw column names and readable labels
        self.feature_labels = {
            "date": "Date",
            "ALLSKY_SFC_SW_DWN": "GHI - Globar Horizontal Irradiance (W/m²)",
            "CLRSKY_SFC_SW_DWN": "GHI Clear Sky - (W/m²)",
            "T2M": "Temperature (C)",
            "RH2M": "Humidity (%)",
            "WS10M": "Wind (m/s)",
            "CLOUD_AMT": "Cloud (%)",
            "PRECTOTCORR": "Precipitation Corrected (mm/day)",
        }

        # Create Data processor to use on inference
        self.data_processor = DataProcessor()

    # ============================================================
    # PAGE 1 — DATA OVERVIEW
    # ============================================================
    def page_data_overview(self):
        st.header("📊 Data Overview")

        st.subheader("🌤️ Raw NASA POWER Dataset Overview")
        st.markdown(
            f"""
        - **Location:** Tamil Nadu, India  
        - **Date Range:** 2025-07-01 → 2026-06-30  
        - **Number of Features:** 8  
        - **Number of Instances:** **{self.raw.shape[0]}**

        ---

        ### Raw NASA POWER Data Preview (meteorological data)
        """
        )

        # Raw Nasa Power Data Set visualization with start and stop box
        col1, col2, col3 = st.columns([1, 1, 7])  # both columns small
        with col1:
            start = st.number_input(
                "Start row", min_value=0, max_value=len(self.raw) - 1, value=0
            )

        with col2:
            end = st.number_input(
                "End row",
                min_value=1,
                max_value=len(self.raw),
                value=5,
            )
        st.dataframe(self.raw.iloc[start:end])

        # Summary statistics of raw data
        st.subheader("Summary Statistics of Raw NASA POWER Data")

        # Summary statistics
        raw_summary = self.raw.describe().transpose()

        # Renomear as features usando o dicionário de labels
        raw_summary.index = [
            self.feature_labels.get(col, col) for col in raw_summary.index
        ]

        # Mostrar no dashboard
        st.dataframe(raw_summary, use_container_width=True)

        st.markdown(
            f"""
        ### 📍 Raw Daily Generation Dataset Details
        - **Location:** Tamil Nadu, India  
        - **Date Range:** 2024-11-24 → 2026-09-15
        - **Number of Features:** 4 
        - **Number of Instances:** **{self.raw2.shape[0]}**
        - **System:** 3 photovoltaic solar panels  
        - **Total installed capacity:** 3 kW  
        - **Approximate panel area:** 15 m²  
        """
        )

        st.subheader("Raw Daily Generation Data Preview")
        col11, col12, col13 = st.columns([1, 1, 7])  # both columns small
        with col11:
            start1 = st.number_input(
                "Start row", min_value=0, max_value=len(self.raw2) - 1, value=0
            )

        with col12:
            end1 = st.number_input(
                "End row",
                min_value=1,
                max_value=len(self.raw2),
                value=5,
            )
        st.dataframe(self.raw2.iloc[start1:end1])

        st.subheader("📘 Training Data Preview (Prepared Data)")
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
                value=5,
            )
        st.dataframe(self.train.iloc[start2:end2])

        st.subheader("📏 Datasets Size:")
        col1, col2, _ = st.columns([2, 2, 3])

        with col1:
            st.markdown(f"""
            - ☀️ **Raw NASA POWER Dataset:** {self.raw.shape[0]} instances 
            - 🟦 **Raw Daily Generation Dataset:** {self.raw2.shape[0]} instances 
            - ⚙️ **Prepared Raw Dataset (after preprocessing):** {self.prepared.shape[0]} instances  
            """)
        with col2:
            st.markdown(f"""
            - 🟢 **Train Dataset:** {self.train.shape[0]} instances  
            - 🟡 **Validation Dataset:** {self.validation.shape[0]} instances  
            - 🔵 **Test Dataset:** {self.test.shape[0]} instances  
            """)

    # ============================================================
    # PAGE 2 — INTERACTIVE EXPLORATION
    # ============================================================
    def page_exploration(self):
        st.header("🔍 Interactive Exploration")

        # --------------------------
        # MAIN LINE CHART
        # --------------------------
        st.markdown(
            "Choose the dates and a feature to visualize its behavior on all graphics below."
        )

        # Use RAW dataset
        df = self.raw.copy()

        # Convert date column if it exists
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

            # --- Date Range Filter (user chooses) ---
            min_date = df["date"].min()
            max_date = df["date"].max()

            col_date1, col_date2, col_3, _ = st.columns([1, 1, 2, 2])

            with col_date1:
                start_date = st.date_input(
                    "Start date", value=min_date, min_value=min_date, max_value=max_date
                )

            with col_date2:
                end_date = st.date_input(
                    "End date", value=max_date, min_value=min_date, max_value=max_date
                )

            # Filter dataframe by user-selected range
            df = df[
                (df["date"] >= pd.to_datetime(start_date))
                & (df["date"] <= pd.to_datetime(end_date))
            ]

            # Avoid empty chart
            if df.empty:
                st.warning("No data available for the selected date range.")
                return

            with col_3:
                # Remove "date" from labels list
                labels_without_date = {
                    key: value
                    for key, value in self.feature_labels.items()
                    if key != "date"
                }

                feature_label = st.selectbox(
                    "Select a feature:", list(labels_without_date.values())
                )

        # Find the corresponding raw column name
        feature = [k for k, v in self.feature_labels.items() if v == feature_label][0]

        # --------------------------
        # FEATURE OVERTIME - LINE CHART
        # --------------------------
        st.markdown("### Feature Overtime")
        # Line chart
        if "date" in df.columns:
            fig = px.line(
                df,
                x="date",
                y=feature,
                title=f"{feature_label} Over Time",
                markers=True,
                labels={
                    "date": "Date",  # Rename axis X
                    feature: feature_label,  # Rename axis Y
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        # --------------------------
        # ROLLING MEAN CHART
        # --------------------------
        st.subheader("Rolling Mean")

        window = st.slider(
            "Select rolling window (days):", min_value=3, max_value=60, value=14
        )

        df["rolling"] = df[feature].rolling(window=window).mean()

        fig2 = px.line(
            df,
            x="date",
            y="rolling",
            title=f"{feature_label} - {window}-day Rolling Mean",
            markers=False,
            labels={"date": "Date", "rolling": f"{feature_label} (Rolling Mean)"},
        )
        st.plotly_chart(fig2, use_container_width=True)

        # --------------------------
        # SCATTER WITH TRENDLINE
        # --------------------------
        st.subheader("Scatter with Trendline")
        fig3 = px.scatter(
            df,
            x="date",
            y=feature,
            trendline="lowess",
            title=f"{feature_label} Scatter with Trendline",
            labels={"date": "Date", feature: feature_label},
        )

        # Make the trendline red
        fig3.update_traces(selector=dict(mode="lines"), line=dict(color="red", width=3))

        st.plotly_chart(fig3, use_container_width=True)

    # ============================================================
    # PAGE 3 — PREDICTION
    # ============================================================

    def page_prediction(self):

        st.header("⚡ Energy Prediction")
        st.subheader("""
            🎯 Goal
            Predict the **daily solar energy generation** for the specific photovoltaic system located in Tamil Nadu (India).

            Adjust the input features below to generate a prediction:
            """)

        # Load RAW dataset (needed for dtypes and default values)
        df_raw = self.raw.copy()

        # RAW columns shown to the user
        raw_cols = df_raw.columns

        input_dict = {}

        # Create 4 columns for inputs
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]

        i = 0  # column rotation counter

        four_decimal_fields = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN"]

        # Create inputs based on RAW dataset
        for col in raw_cols:
            current_col = cols[i % 4]
            i += 1

            # Friendly label
            label = self.feature_labels.get(col, col)

            # Feature DATE → use calendar input
            if col == "date":
                with current_col:
                    input_dict[col] = st.date_input(label)
                continue

            # Numeric input → use mean as default
            if df_raw[col].dtype != "object":
                default_value = float(df_raw[col].mean())

                # Set the decimal
                if col in four_decimal_fields:
                    fmt = "%.4f"
                else:
                    fmt = "%.2f"

                with current_col:
                    input_dict[col] = st.number_input(
                        label, value=default_value, format=fmt
                    )

        # Convert dict → DataFrame
        input_df = pd.DataFrame([input_dict])

        # Convert date to datetime
        input_df["date"] = pd.to_datetime(input_df["date"])

        # Rename RAW → model columns
        input_df = self.data_processor.rename_features_df(input_df)

        # Create derived features (same logic as training)
        input_df = self.data_processor.create_features_df(input_df)

        # Create seasons (same logic as training)
        input_df = self.data_processor.encode_season_df(input_df)

        # Ensure model receives exactly the columns it expects
        final_df = input_df[self.predictor.model.feature_names_in_]

        # Predict
        col_btn, col_result, _ = st.columns([1, 2, 11])

        with col_btn:
            predict_clicked = st.button("Predict")

        with col_result:
            if predict_clicked:
                pred = self.predictor.predict(final_df)
                st.success(f"Predicted Energy: {pred:.2f} kWh")

        # GRAPHIC FOR PREDICT VALUE
        if predict_clicked:
            max_value = (
                self.prepared["generated_energy"].max() * 1.5
            )  # set the max value = data_set_prepared - generated_energy + 50%
            _, col_graph, _ = st.columns([1, 2, 11])

            with col_graph:
                figA = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=pred,
                        gauge={
                            "axis": {"range": [0, max_value]},
                            "bar": {"color": "orange"},
                            "steps": [
                                {"range": [0, max_value * 0.5], "color": "#d0e1f9"},
                                {
                                    "range": [max_value * 0.5, max_value],
                                    "color": "#a8c4f0",
                                },
                            ],
                        },
                    )
                )

                figA.update_layout(
                    height=150,  # graph height
                    width=150,  # graph width
                    margin=dict(l=10, r=10, t=10, b=10),  # opcional: margin
                )

                st.plotly_chart(figA, use_container_width=True)

    # ============================================================
    # MAIN NAVIGATION
    # ============================================================
    def run(self):
        st.set_page_config(layout="wide")

        st.title("🔆 Solar Energy Production Dashboard")

        # 🔧 CSS for navbar on top of the page
        st.markdown(
            """
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
            margin-top: 0.2rem !important;   /* space above the title */
            margin-bottom: 0.5rem !important; /* space below the title */

        }

        /* Push down the content */
        .main {
            margin-top: 10px !important;
        }

        /* Remove lateral spaces */
        .block-container {
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Top Menu
        menu = st.radio(
            "",
            [
                "📊 Data Overview",
                "🔍 Interactive Exploration",
                "⚡ Prediction",
                "📉 Model Performance",
            ],
            horizontal=True,
        )

        #  st.markdown("---")

        # Menu
        if menu == "📊 Data Overview":
            self.page_data_overview()
        elif menu == "🔍 Interactive Exploration":
            self.page_exploration()
        elif menu == "⚡ Prediction":
            self.page_prediction()
        elif menu == "📉 Model Performance":
            self.page_performance()


# ============================================================
# RUN DASHBOARD
# ============================================================
if __name__ == "__main__":
    Dashboard().run()
