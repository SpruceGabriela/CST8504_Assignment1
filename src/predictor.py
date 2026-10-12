import joblib

# import pandas as pd - update when get real dataframe data


class Predictor:
    def __init__(
        self, model_path="models/best_model.pkl", scaler_path="models/scaler.pkl"
    ):
        """Loads the trained model and scaler from their paths"""

        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Ensure the model was saved during training."
            ) from e

        # Loads the scaler if it exists, or throw an error if it doesn't
        try:
            self.scaler = joblib.load(scaler_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Scaler file not found at {scaler_path}. Run the data pipeline first."
            ) from e

    def predict(self, input_data):
        """
        Makes a prediction based on the input data, a DataFrame with the same structure as the training data
        """

        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)

        [predicted_kwh] = prediction  # get the first value

        # Returns the predicted numeric value, accessing the first element of the array
        # Energy can't be negative, so we return 0 if the prediction is negative
        if predicted_kwh < 0:
            return 0.0
        return float(predicted_kwh)
