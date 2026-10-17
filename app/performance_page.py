import pandas as pd
import plotly.express as px
import streamlit as st


class PerformancePage:
    def __init__(self, models_dir="models"):
        # Load files created during training (model_trainer.py)
        self.metrics = pd.read_csv(
            f"{models_dir}/model_metrics.csv", index_col="dataset"
        )
        self.predictions = pd.read_csv(f"{models_dir}/test_predictions.csv")
        self.coefficients = pd.read_csv(f"{models_dir}/coefficients.csv")

    def render_page(self):
        """Draws the performance page.
        The dashboard only needs to call this method."""
        st.header("📈 Model Performance")
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
        """Scatter plot: actual kWh (x) vs predicted kWh (y), with the y = x line."""
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

        st.plotly_chart(fig)
        st.caption(
            "Each dot is one day the model had never seen. If the prediction was perfect, the dot sits "
            "on the dashed line. Dots below the line are days the model expected more energy than the plant "
            "actually produced."
        )

    def show_residuals(self):
        """Scatter plot: predicted kWh (x) vs residual (y), with a line at zero."""
        st.subheader("📉 Residuals")
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


if __name__ == "__main__":
    PerformancePage().render_page()
