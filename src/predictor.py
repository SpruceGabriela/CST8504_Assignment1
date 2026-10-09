import joblib
import pandas as pd


class Predictor:
    def __init__(
        self, model_path="models/best_model.pkl", scaler_path="models/best_scaler.pkl"
    ):
        """Loads the trained model and optional scaler from their paths"""

        self.model = joblib.Load(model_path)

        # Loads the scale if it exists, or sets it to none
        try:
            self.scaler = joblib.Load(scaler_path)
        except FileNotFoundError:
            self.scaler = None
            print(
                "An error occured tryin to load the scaler, make sure the file exists and the path is correct"
            )

    def predict(self, input_data):
        """
        Makes a prediction based on the input data, a DataFrame with the same structure as the training data
        """

        # If scaler was loaded, scale the input data in the same way as training data
        if self.scaler is not None:
            input_scaled = self.scaler.transform(input_data)
            prediction = self.model.predict(input_scaled)
        else:
            prediction = self.model.predict(input_data)

        # Returns the predicted numeric value, accessing the first element of the array
        return float(prediction[0])
