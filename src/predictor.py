import joblib


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
