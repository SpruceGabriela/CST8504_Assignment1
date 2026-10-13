import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.errors = None

    def load_rawdata(self):  # Load dataset
        self.df = pd.read_csv(self.filepath)
        return self

    def rename_features(self):
        self.df = self.df.rename(
            columns={
                "ALLSKY_SFC_SW_DWN": "ghi",
                "CLRSKY_SFC_SW_DWN": "ghi_clear",
                "ALLSKY_SFC_SW_DNI": "dni",
                "T2M": "temp",
                "RH2M": "humidity",
                "WS10M": "wind",
                "CLOUD_AMT": "clouds",
                "PRECTOTCORR": "rain",
            }
        )
        return self

    def format_date(self):
        self.df["date"] = pd.to_datetime(self.df["date"], format="%Y%m%d")
        return self

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self

    def generate_id(self):
        self.df["id"] = range(1, len(self.df) + 1)
        # Move ID column as 1st column of dataset
        self.df = self.df[["id"] + [col for col in self.df.columns if col != "id"]]
        return self

    def replace_missing(self):
        # Replace NASA_POWER missing values (-999.0) with NaN
        self.df = self.df.replace(-999.0, np.nan)

        # Interpolate missing values (best for time series)
        self.df = self.df.interpolate(method="linear")
        return self

    # Create new features ghi_clear_ratio, day_of_year, month, season
    def create_features(self):
        # feature ghi_clear_ratio = real irradiance / clear-sky irradiance
        self.df["ghi_clear_ratio"] = self.df["ghi"] / self.df["ghi_clear"]
        self.df["day_of_year"] = self.df["date"].dt.dayofyear
        self.df["month"] = self.df["date"].dt.month
        self.df["season"] = self.df["month"].apply(self.get_season)
        return self

    @staticmethod
    def get_season(month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "summer"
        elif month in [6, 7, 8, 9]:
            return "monsoon"  # daily heavy rain
        else:
            return "post_monsoon"

    # Outliers detection (IQR method)
    def detect_outliers(self):
        # Select columns with numeric data types and pick the columns names to use on next line
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        # Apply the function iqr_outliers on numeric columns
        flags = self.df[numeric_cols].apply(self.iqr_outliers)
        df_outliers = self.df[
            flags.any(axis=1)
        ].copy()  # verify every line, and select/copy instance where the flag for outlier is true
        df_outliers["classification"] = df_outliers.apply(self.classify_outlier, axis=1)

        # errors receive OUTLIERS CLASSIFIED AS SENSOR ERROR
        self.errors = df_outliers[
            df_outliers["classification"] == "possible sensor error"
        ]
        return self

    @staticmethod
    def iqr_outliers(series):
        Q1 = series.quantile(0.25)  # values on 1st quarter
        Q3 = series.quantile(0.75)  # values on 3rd quarter
        IQR = Q3 - Q1  # range of the middle half of the dataset
        lower = Q1 - 1.5 * IQR  # lower limit
        upper = Q3 + 1.5 * IQR  # upper limit
        return (series < lower) | (series > upper)

    # Outliers classification
    @staticmethod
    def classify_outlier(row):
        # Real events, not outliers
        if row["rain"] > 50:
            return "heavy monsoon rain (real)"
        if row["clouds"] > 90:
            return "dense monsoon clouds (real)"
        if row["temp"] < 5:
            return "cold wave (real)"
        if row["temp"] > 40:
            return "extreme hot (real)"
        # Possible sensor errors
        if row["ghi"] < 0.5 and row["clouds"] < 20:
            return "possible sensor error"
        return "normal meteorological outlier"

    def remove_errors(self):
        # copy all instances that don't have error (ist's not an outlier)
        self.df = self.df[~self.df.index.isin(self.errors.index)].copy()
        return self

    # One-hot enconding for feature season
    def encode_season(self):
        # Create one-hote enconding from feature season, starting with 's'
        self.df = pd.get_dummies(
            self.df, columns=["season"], prefix="s", drop_first=False
        )
        # Create list with all columns starting with s_
        dummy_cols = [col for col in self.df.columns if col.startswith("s_")]
        # Change the type from Str (True/False) to int (1/0)
        self.df[dummy_cols] = self.df[dummy_cols].astype(int)
        return self

    def load_real_generation(self, kaggle_path):
        """
        Loads real solar generation from Kaggle dataset (Daily Generation.xlsx)
        and prepares it for merging with the NASA dataset.
        """
        df_kaggle = pd.read_csv(kaggle_path)

        # Rename columns to match internal naming
        df_kaggle = df_kaggle.rename(
            columns={"Date": "date", "Cleaned Energy (kWh)": "generated_energy"}
        )

        # Convert date to datetime
        df_kaggle["date"] = pd.to_datetime(df_kaggle["date"])

        # Store for later merge
        self.df_kaggle = df_kaggle

        return self

    def merge_real_label(self):
        """
        Merges real cleaned energy values into the main dataset.
        """
        # Ensure date is datetime
        self.df["date"] = pd.to_datetime(self.df["date"])

        # Merge by date
        self.df = self.df.merge(
            self.df_kaggle[["date", "generated_energy"]], on="date", how="left"
        )

        # Replace label with real cleaned energy
        # self.df["label"] = self.df["cleaned_energy"]

        return self

    # Normalize all numeric features
    def normalize(self):
        # List of numeric features to be normalized
        cols_to_scale = [
            "ghi",
            "ghi_clear",
            "temp",
            "humidity",
            "wind",
            "clouds",
            "rain",
            "day_of_year",
            "month",
            "ghi_clear_ratio",
        ]
        scaler = MinMaxScaler()
        self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])
        return self

    def split_data(self, label_col="generated_energy", seed=42):
        """
        Splits the dataset into train (70%), validation (15%), and test (15%)
        using a fixed random seed for reproducibility.
        """

        # Separate features and label
        X = self.df.drop(columns=[label_col])
        y = self.df[label_col]

        # Train (70%) + Temp (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=seed,  # division of data_set will be always the same, will not generate diferent splits
        )

        # Validation (15%) + Test (15%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=seed
        )

        # Store splits inside the object
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        return self

    def save_splits(self, output_dir="data/processed"):
        """
        Saves train, validation and test splits as separate CSV files.
        Requires split_data() to be executed first.
        """

        # Ensure directory exists
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Save each split
        self.X_train.assign(label=self.y_train).to_csv(
            f"{output_dir}/train.csv", index=False
        )
        self.X_val.assign(label=self.y_val).to_csv(
            f"{output_dir}/validation.csv", index=False
        )
        self.X_test.assign(label=self.y_test).to_csv(
            f"{output_dir}/test.csv", index=False
        )

        return self

    # Create csv documento with prepared dataset (whole data set withou splits)
    def save_csv(self, output_path):
        self.df.to_csv(output_path, index=False)
        return self


processor = DataProcessor("data/raw/data_set_raw.csv")

processor.load_rawdata().rename_features().format_date().remove_duplicates().generate_id().replace_missing().create_features().detect_outliers().remove_errors().encode_season().load_real_generation(
    "data/raw/daily_generation.csv"
).merge_real_label().normalize().split_data().save_splits().save_csv(
    "data/processed/data_set_prepared.csv"
)
