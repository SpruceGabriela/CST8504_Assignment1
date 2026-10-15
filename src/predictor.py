import joblib
import pandas as pd

from src.data_processor import DataProcessor


class Predictor:
    def __init__(
        self, model_path="models/linear_regression.pkl", scaler_path="models/scaler.pkl"
    ):
        """
        Loads the trained model and scaler from their paths.

        Raises FileNotFoundError if either file is missing.
        """

        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Ensure the model was saved during training."
            ) from e

        # Loads the scaler if it exists
        try:
            self.scaler = joblib.load(scaler_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Scaler file not found at {scaler_path}. Run the data pipeline first."
            ) from e

    def predict(self, input_data):
        """
        Predicts the daily energy generation for a single day.

        input_data: DataFrame with exactly one row and the unscaled feature columns
                    (the same column names as the processed dataset).
        Returns: predicted energy in kWh, never negative.
        """

        # Column selection/scaling logic refactored with help from AI assistant - Claude
        # First I was using the original input_data directly, but it was scaling all columns

        copied_input_data = input_data.copy()  # Avoid modifying original data
        scaled_columns = self.scaler.feature_names_in_

        features_to_scale = copied_input_data[scaled_columns]
        copied_input_data[scaled_columns] = self.scaler.transform(features_to_scale)

        model_input = copied_input_data[self.model.feature_names_in_]
        prediction = self.model.predict(model_input)

        # Destructure array to get the first value, accessing the first element of the array
        [predicted_kwh] = prediction

        # Returns the predicted numeric value
        # Energy can't be negative, so we return 0 if the prediction is negative
        if predicted_kwh < 0:
            return 0.0
        return float(predicted_kwh)

    # Used by the dashboard prediction page: it will receive the raw values typed by the user.
    # The assignment PDF warns that the input from the dashboard has
    # to go through exactly the same preprocessing as your training data, or the predictions
    # will be nonsense. The dashboard just calls this method and shows the result.
    def predict_from_inputs(
        self, ghi, ghi_clear, temp, humidity, wind, clouds, rain, date
    ):
        """
        Predicts the daily energy generation from the raw values typed in the dashboard.
        ghi, ghi_clear: kWh/m²/day | temp: °C | humidity, clouds: % | wind: m/s | rain: mm/day
        date: the day to predict, e.g. "2026-03-15" or a datetime.date
        Returns: predicted energy in kWh, never negative.
        """

        # ghi_clear must be greater than 0 to avoid division by zero when calculating ghi_clear_ratio
        if ghi_clear <= 0:
            raise ValueError("ghi_clear must be greater than 0.")

        date = pd.to_datetime(date)

        # Same derived features as DataProcessor.create_features()
        input_data = pd.DataFrame(
            [
                {
                    "ghi": ghi,
                    "ghi_clear": ghi_clear,
                    "temp": temp,
                    "humidity": humidity,
                    "wind": wind,
                    "clouds": clouds,
                    "rain": rain,
                    "ghi_clear_ratio": ghi / ghi_clear,
                    "day_of_year": date.dayofyear,
                    "month": date.month,
                }
            ]
        )

        # Season one-hot columns, as in DataProcessor.encode_season():
        # 1 for the season of this date, 0 for the others
        season = DataProcessor.get_season(date.month)
        for column in self.model.feature_names_in_:
            if column.startswith("s_"):
                input_data[column] = int(column == f"s_{season}")

        return self.predict(input_data)
