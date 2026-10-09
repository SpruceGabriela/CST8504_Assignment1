import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelTrainer:
    def __init__(self):
        # Define both regression models inside the constructor
        # to ensure initialization happens when the class is instantiated
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        }
        self.results = {}  # dictionary to store the results of each model
        self.best_model = None
        self.best_model_name = ""

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate both models, storing their performance metrics.
        """

        # Initializing the best R2 score (compares the model's predictions to the data's average)
        # as negative infinity to ensure any model's score will be higher
        best_r2 = float("-inf")

        for name, model in self.models.items():
            # Measuring the time took to train the model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Make predictions
            y_prediction = model.predict(X_test)

            # Metrics
            mae = mean_absolute_error(y_test, y_prediction)
            rmse = (
                mean_squared_error(y_test, y_prediction) ** 0.5
            )  # RMSE calculation (Root Mean Squared Error)
            r2 = r2_score(y_test, y_prediction)

            # Store results in the dictionary defined in the constructor
            self.results[name] = {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "training_time": training_time,
            }

            # Update best model if has the highest R2 score
            if r2 > best_r2:
                best_r2 = r2
                self.best_model = model
                self.best_model_name = name

        # Converts the results of the dictiornary to Pandas DataFrame, saves CSV and returns DataFrame for StreamLit
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv("models/model_comparison.csv")
        return results_df

    def save_best_model(self, file_path="models/best_model.pkl"):
        """
        Saves the best model to a file in the folder /models.
        """
        if self.best_model is not None:
            joblib.dump(self.best_model, file_path)
            print(f"Best model was save to {file_path}")
        else:
            print("No model has been trained yet, nothing to save")
