import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Ensure Python can find the src/ package
# ------------------------------------------------------------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processor import DataProcessor
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

        # Load files created during training (model_trainer.py)
        self.metrics = pd.read_csv("models/model_metrics.csv", index_col="dataset")
        self.predictions = pd.read_csv("models/test_predictions.csv")
        self.coefficients = pd.read_csv("models/coefficients.csv")

        # Load model + scaler
        self.predictor = Predictor(
            model_path="models/linear_regression.pkl", scaler_path="models/scaler.pkl"
        )

        # Page settings
        st.set_page_config(
            page_title="Solar Energy Production Dashboard",
            page_icon="☀️",
            layout="wide",
        )

        # Create a mapping between raw column names and readable labels
        self.feature_labels = {
            "date": "Date",
            "ALLSKY_SFC_SW_DWN": "GHI - Globar Horizontal Irradiance (W/m²)",
            "CLRSKY_SFC_SW_DWN": "GHI Clear Sky - (W/m²)",
            "T2M": "Temperature (C)",
            "RH2M": "Humidity (%)",
            "WS2M": "Wind (m/s)",
            "CLOUD_AMT": "Cloud (%)",
            "PRECTOTCORR": "Precipitation Corrected (mm/day)",
        }

        # Create Data processor to use on inference
        self.data_processor = DataProcessor()

    # ============================================================
    # PAGE 1 — DATA OVERVIEW
    # ============================================================
    def page_data_overview(self):
        """Draws the Data Overview page.
        The dashboard only needs to call this method."""
        st.header("📊 Data Overview")

        self.raw_nasa_overview()
        self.raw_nasa_preview()
        self.raw_nasa_stats()
        self.raw_generation_overview()
        self.raw_generation_preview()
        self.train_preview()
        self.dataset_size()

    def raw_nasa_overview(self):
        """Explore the Raw Nasa Power data set.
        - Details about the Dataset
        """
        st.subheader("🌤️ Raw NASA POWER Dataset Overview")
        st.markdown(
            f"""
        - **Location:** Tamil Nadu, India  
        - **Date Range:** 2025-07-01 → 2026-06-30  
        - **Number of Features:** 8  
        - **Number of Instances:** {self.raw.shape[0]}

        ---

        ### Raw NASA POWER Data Preview (meteorological data)
        """
        )

    def raw_nasa_preview(self):
        """
        Raw Nasa Power Data Set visualization with start and end row box
        """
        col1, col2, _col3 = st.columns([1, 1, 7])  # both columns small
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

    def raw_nasa_stats(self):
        """
        Summary statistis of the raw Nasa Power Dataset
        """
        st.subheader("Summary Statistics of Raw NASA POWER Data")

        # Summary statistics
        raw_summary = self.raw.describe().transpose()

        # Rename features using labels dictionary
        raw_summary.index = [
            self.feature_labels.get(col, col) for col in raw_summary.index
        ]

        # Show in dashboard
        st.dataframe(raw_summary, use_container_width=True)

    def raw_generation_overview(self):
        """
        Explore the Raw Daily Generation data set.
        - Details about the Dataset
        """
        st.markdown(
            f"""
        ### 📍 Raw Daily Generation Dataset Overview
        - **Location:** Tamil Nadu, India  
        - **Date Range:** 2024-11-24 → 2026-09-15
        - **Number of Features:** 4 
        - **Number of Instances:** {self.raw2.shape[0]}
        - **System:** 3 photovoltaic solar panels  
        - **Total installed capacity:** 3 kW  
        - **Approximate panel area:** 15 m²  
        """
        )

    def raw_generation_preview(self):
        """
        Raw Daily Generation Data Set visualization with start and end row box
        """
        st.subheader("Raw Daily Generation Data Preview")
        col11, col12, _col13 = st.columns([1, 1, 7])  # both columns small
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

    def train_preview(self):
        """
        Training Data Set visualization with start and end row box
        """
        st.subheader("📘 Training Data Preview (Prepared Data)")
        col111, col112, _col113 = st.columns([1, 1, 5])  # both columns small
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

    def dataset_size(self):
        """
        Describe datasets size.
        """

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
        """Draws the exploration page.
        The dashboard only needs to call this method."""
        st.header("🔍 Interactive Exploration")
        st.markdown(
            "Choose the dates and a feature to visualize its behavior on all graphics below."
        )

        self.exploration_options()
        self.overtime()
        self.rolling_mean()
        self.scatter()

    def exploration_options(self):
        """
        Builds option selector. Ensures the user selects a date range and a feature, and then displays all charts based on those inputs.
        """
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

        # Save for other methods
        self.df = df
        self.feature = feature
        self.feature_label = feature_label

    def overtime(self):
        """
        Plot atime-series visualization (Line Chart)
        """
        st.markdown("### Feature Overtime")
        # Line chart
        if "date" in self.df.columns:
            fig = px.line(
                self.df,
                x="date",
                y=self.feature,
                title=f"{self.feature_label} Over Time",
                markers=True,
                labels={
                    "date": "Date",  # Rename axis X
                    self.feature: self.feature_label,  # Rename axis Y
                },
            )
            st.plotly_chart(fig, use_container_width=True)

    def rolling_mean(self):
        """
        Plots the rolling mean of the selected feature.
        Allows the user to choose a rolling window (3–60 days),
        and visualizes the smoothed trend to highlight medium‑term behavior.
        """
        st.subheader("Smoothed Trend (Rolling Mean)")

        window = st.slider("Select rolling window (days):", 3, 60, 14)

        self.df["rolling"] = self.df[self.feature].rolling(window=window).mean()

        fig2 = px.line(
            self.df,
            x="date",
            y="rolling",
            title=f"{self.feature_label} - {window}-day Rolling Mean",
            markers=False,
            labels={"date": "Date", "rolling": f"{self.feature_label} (Rolling Mean)"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    def scatter(self):
        """
        Shows a scatter plot with a LOWESS trendline.
        Helps identify patterns, noise, and non‑linear relationships in the selected feature over time.
        The trendline is styled in red for emphasis.
        """
        st.subheader("Feature Trend & Variability")
        fig3 = px.scatter(
            self.df,
            x="date",
            y=self.feature,
            trendline="lowess",
            title=f"{self.feature_label} Scatter with Trendline",
            labels={"date": "Date", self.feature: self.feature_label},
        )

        # Make the trendline red
        fig3.update_traces(selector=dict(mode="lines"), line=dict(color="red", width=2))

        st.plotly_chart(fig3, use_container_width=True)

    # ============================================================
    # PAGE 3 — PREDICTION
    # ============================================================

    def page_prediction(self):
        """
        Interactive module for predicting daily solar energy output based on model and user inputs.
        """

        st.header("⚡ Energy Prediction")
        st.subheader("""
            🎯 Goal
            Predict the **daily solar energy generation** for the specific photovoltaic system located in Tamil Nadu (India).

            Adjust the input features below to generate a prediction:
            """)

        self.user_inputs()
        self.user_prediction()

    def user_inputs(self):
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

        # Save for other methods
        self.input_df = input_df

    def user_prediction(self):

        # Rename RAW → model columns
        self.input_df = self.data_processor.rename_features_df(self.input_df)

        # Create derived features (same logic as training)
        self.input_df = self.data_processor.create_features_df(self.input_df)

        # Create seasons (same logic as training)
        self.input_df = self.data_processor.encode_season_df(self.input_df)

        # Ensure model receives exactly the columns it expects
        final_df = self.input_df[self.predictor.model.feature_names_in_]

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
            # set the max value = data_set_prepared - generated_energy + 50%
            max_value = self.prepared["generated_energy"].max() * 1.5
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
    # PAGE 4 — PERFORMANCE
    # ============================================================

    def page_performance(self):
        """Draws the performance page.
        The dashboard only needs to call this method."""
        st.header("📉 Model Performance")
        st.markdown(
            "Linear Regression trained on 70% of the days, checked on 15% (validation) "
            "and evaluated once on the other 15% for testing, which the model never saw."
        )

        self.show_metrics()
        self.show_predicted_vs_actual()
        self.show_residuals()
        self.show_coefficients()
        self.show_worst_days()

    def show_metrics(self):
        """Table with MAE, RMSE and R² on the train, validation and test sets."""
        st.subheader("📊 Metrics")
        st.dataframe(self.metrics.round(3))
        st.caption(
            "MAE is the average error in kWh per day, and RMSE is similar but gives more weight to big misses "
            "(for both, lower is better). R² shows how much of the day-to-day change in energy the model explains "
            "(1 is perfect, 0 is no better than guessing the average). The test row is the final score."
        )

    def show_predicted_vs_actual(self):
        """Line chart: actual and predicted kWh for each test day, in date order."""

        st.subheader("🎯 Predicted vs Actual")

        predictions = self.predictions.sort_values("date")

        fig = px.line(
            predictions,
            x="date",
            y=["actual", "predicted"],
            markers=True,
            hover_data=["date"],
            labels={
                "date": "Date",
                "value": "Energy (kWh)",
                "variable": "",
            },
            title="Predicted vs actual energy (test set)",
            color_discrete_map={"actual": "#1ABC9C", "predicted": "#D2457D"},
        )
        fig.update_yaxes(range=[0, 25], dtick=2)
        st.plotly_chart(fig)
        st.caption(
            "Turquoise is the real energy and pink is the model's prediction, for days the model never saw. "
            "The closer the lines, the better. Big gaps are days the plant produced less than expected."
        )

    def show_residuals(self):
        """Scatter plot: predicted kWh (x) vs residual (y), with a line at zero."""
        st.subheader("📈 Residuals")
        fig = px.scatter(
            self.predictions,
            x="predicted",
            y="residual",
            hover_data=["date"],
            labels={
                "predicted": "Predicted energy (kWh)",
                "residual": "Residual: actual - predicted (kWh)",
            },
            title="Residuals (test set)",
        )
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig)
        st.caption(
            "A residual is the miss for one day (actual minus predicted). Dots scattered evenly around the dashed "
            "zero line mean the model has no systematic bias. The few dots far below zero are days the model "
            "overestimated by a lot, most likely days when the plant wasn't working."
        )

    def show_coefficients(self):
        """Bar chart: which features push the prediction up and which pull it down."""
        st.subheader("⚖️ Which features push the prediction up and which pull it down")
        # Sorting from the most negative to the most positive, so the bars are in order
        coefficients = self.coefficients.sort_values("coefficient")
        fig = px.bar(
            coefficients,
            x="coefficient",
            y="feature",
            orientation="h",
            labels={
                "coefficient": "Coefficient (effect on kWh, scaled features)",
                "feature": "Feature",
            },
            title="Linear Regression coefficients",
        )
        st.plotly_chart(fig)
        st.caption(
            "Bars to the right increase the predicted energy. Bars to the left decrease it. "
            "Features are scaled from 0 to 1, so bar sizes can be compared with each other: "
            "solar irradiance (ghi) has by far the biggest effect."
        )

    def show_worst_days(self):
        """Table with the test days where the model made the biggest overestimations."""
        st.subheader("🔍 Days the model overestimated the most")
        worst_days = self.predictions.sort_values("residual").head(5)
        worst_days = worst_days[["date", "actual", "predicted", "residual"]].round(2)
        st.dataframe(worst_days, hide_index=True)
        st.caption(
            "The 5 days where the model overestimated the most. On these days there was plenty of sun, but the "
            "plant produced almost nothing, which suggests it was offline. No weather data can predict that, so "
            "these errors come from the data, not from the model."
        )

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
            z-index: 99999;
            background-color: light-dark(#ffffff, #0e1117) !important;
            padding: 15px 5px 20px 0px;
            width: 100vw;
            border-bottom: 1px solid #ccc;
            margin-top: 0.2rem !important;   /* space above the title */
            margin-bottom: 0.5rem !important; /* space below the title */

        }

        /* Push down the content */
        .main {
            margin-top: 70px !important;
        }

        /* Remove lateral spaces */
        .block-container {
            padding-top: 7.5rem !important;
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
