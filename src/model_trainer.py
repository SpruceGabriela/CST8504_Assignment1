import time

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_COLUMN = "label"  # Feature that we want to predict (ghi)

# Columns that shouldn't be used as features for training the model
NON_FEATURE_COLUMNS = [
    "id",  # Just the ID of the row
    "date",  # Just text, not a feature
    TARGET_COLUMN,  # It's the answer, so we can't use
]


# Logic and order of events done with help from AI assistant - Claude
class ModelTrainer:
    def __init__(self, data_dir="data/processed", models_dir="models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        # Define linear regression model inside the constructor
        # to ensure initialization happens when the class is instantiated
        self.model = LinearRegression()
        self.is_trained = False
        self.training_time = None
        self.metrics = {}

    def load_data(self):
        """Loads the train, validation and test splits and
        separates the features (X) from the label (y).
        """

        # Load train, validation and test splits from CSV
        self.train_df = pd.read_csv(f"{self.data_dir}/train.csv")
        self.validation_df = pd.read_csv(f"{self.data_dir}/validation.csv")
        self.test_df = pd.read_csv(f"{self.data_dir}/test.csv")

        # Split features and label for each dataset (method defined below)
        self.X_train, self.y_train = self.split_features_and_label(self.train_df)
        self.X_validation, self.y_validation = self.split_features_and_label(
            self.validation_df
        )
        self.X_test, self.y_test = self.split_features_and_label(self.test_df)
        return self

    def split_features_and_label(self, df):
        """
        Splits the input DataFrame into features (X) and label (y).
        """

        # Every column except id, date and label is a feature
        x = df.drop(columns=NON_FEATURE_COLUMNS)
        y = df[TARGET_COLUMN]
        return x, y

    def train_model(self):
        """
        Trains the linear regression model using the training data.
        Measures the time taken for training and sets the is_trained flag.
        """

        start_time = time.perf_counter()
        self.model.fit(self.X_train, self.y_train)  # Training model
        self.training_time = time.perf_counter() - start_time
        self.is_trained = True
        return self

    def evaluate_model(self, x, y):
        """
        Returns MAE, RMSE and R2 of the model on the given data.
        """
        y_pred = self.model.predict(x)  # Predicting the label using the trained model
        return {
            # Mean Absolute Error - average of errors between predicted and actual values
            "MAE": mean_absolute_error(y, y_pred),
            # Root Mean Squared Error - square root of the average of squared differences between predicted and actual values
            "RMSE": mean_squared_error(y, y_pred) ** 0.5,
            # R2: how well the model explains the data (1 = perfect, 0 = same as predicting the average)
            "R2": r2_score(y, y_pred),
        }

    def evaluate_train_and_validation(self):
        """
        Evaluates the trained model on both training and validation datasets.
        Similar train and validation metrics means the model is generalizing well,
        while a large difference indicates overfitting.
        """
        self.metrics["train"] = self.evaluate_model(self.X_train, self.y_train)
        self.metrics["validation"] = self.evaluate_model(
            self.X_validation, self.y_validation
        )
        return self

    def evaluate_test(self):
        """
        Evaluates the trained model on the test dataset.
        The test dataset is used only once, for the final result
        """
        self.metrics["test"] = self.evaluate_model(self.X_test, self.y_test)
        return self

    def save_model(self):
        """
        Saves the model, the metrics, the coefficients and
        the test predictions in the models folder
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before saving. Call train_model() first."
            )
        joblib.dump(self.model, f"{self.models_dir}/linear_regression.pkl")

        # Define metrics table for the performance page
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df["training_time_s"] = self.training_time
        metrics_df.to_csv(f"{self.models_dir}/model_metrics.csv", index_label="dataset")

        # Positive coefficients push the prediction up, negative ones pull it down
        coefficients_df = pd.DataFrame(
            {"feature": self.X_train.columns, "coefficient": self.model.coef_}
        ).sort_values("coefficient", ascending=False)
        coefficients_df.to_csv(f"{self.models_dir}/coefficients.csv", index=False)

        # Used by the performance page: predicted vs actual and residuals
        test_predictions_df = pd.DataFrame(
            {
                "date": self.test_df["date"],
                "actual": self.y_test,
                "predicted": self.model.predict(self.X_test),
            }
        )
        test_predictions_df["residual"] = (
            test_predictions_df["actual"] - test_predictions_df["predicted"]
        )
        test_predictions_df.to_csv(
            f"{self.models_dir}/test_predictions.csv", index=False
        )

        return self


if __name__ == "__main__":
    # Calling the ModelTrainer class and executing pipeline
    trainer = ModelTrainer()
    (
        trainer.load_data()
        .train_model()
        .evaluate_train_and_validation()
        .evaluate_test()
        .save_model()
    )

    # Printing the results for a friendly console output
    metrics_df = pd.DataFrame(trainer.metrics).T.round(3)
    print("Linear Regression results (MAE and RMSE in kWh):")
    print(metrics_df)
    print(f"Training time: {trainer.training_time:.4f} s")
    print(f"Files saved to {trainer.models_dir}/")
