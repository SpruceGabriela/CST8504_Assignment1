import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.errors = None

    def load_rawdata(self): # Load dataset
        self.df = pd.read_csv(self.filepath)
        return self

    def rename_features(self):
        self.df = self.df.rename(columns={
            "ALLSKY_SFC_SW_DWN": "ghi",
            "CLRSKY_SFC_SW_DWN": "ghi_clear",
            "ALLSKY_SFC_SW_DNI": "dni",
            "T2M": "temp",
            "RH2M": "humidity",
            "WS10M": "wind",
            "CLOUD_AMT": "clouds",
            "PRECTOTCORR": "rain"
        })
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
        if month in [12, 1, 2]: return "winter"
        elif month in [3, 4, 5]: return "spring"
        elif month in [6, 7, 8]: return "summer"
        else: return "fall"

    # Outliers detection (IQR method)
    def detect_outliers(self):
        # Select columns with numeric data types and pick the columns names to use on next line
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        # Apply the function iqr_outliers on numeric columns
        flags = self.df[numeric_cols].apply(self.iqr_outliers)
        df_outliers = self.df[flags.any(axis=1)].copy() # verify every line, and select/copy instance where the flag for outlier is true
        df_outliers["classification"] = df_outliers.apply(self.classify_outlier, axis=1)

        # errors receive OUTLIERS CLASSIFIED AS SENSOR ERROR
        self.errors = df_outliers[df_outliers["classification"] == "possible sensor error"]
        return self

    @staticmethod
    def iqr_outliers(series):
        Q1 = series.quantile(0.25) # values on 1st quarter
        Q3 = series.quantile(0.75) # values on 3rd quarter
        IQR = Q3 - Q1 # range of the middle half of the dataset
        lower = Q1 - 1.5 * IQR # lower limit
        upper = Q3 + 1.5 * IQR # upper limit
        return (series < lower) | (series > upper)

    # Outliers classification
    @staticmethod
    def classify_outlier(row):
        # Real events, not outliers
        if row["rain"] > 15:
            return "heavy rain (real)"
        if row["clouds"] > 95:
            return "dense clouds (real)"
        if row["temp"] < -25:
            return "extreme cold (real)"
        if row["temp"] > 35:
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
        self.df = pd.get_dummies(self.df, columns=["season"], prefix="s", drop_first=False)
        # Create list with all columns starting with s_
        dummy_cols = [col for col in self.df.columns if col.startswith("s_")]
        # Change the type from Str (True/False) to int (1/0)
        self.df[dummy_cols] = self.df[dummy_cols].astype(int)
        return self
    
    def create_ghi_generation(self, area=1.7, efficiency=0.20, PR=0.75):
        """
        Estimates PV energy generation using only GHI (no POA).
        Useful when POA or DHI is unavailable or PVLib causes errors.
        """
        # Energy in Wh
        self.df["energy_ghi"] = self.df["ghi"] * area * efficiency * PR

        # Convert to kWh
        self.df["energy_ghi_kWh"] = self.df["energy_ghi"] / 1000

        return self

    def create_label(self):
        self.df["label"] = self.df["energy_ghi_kWh"]
        return self
    
    # Normalize all numeric features
    def normalize(self):
        # List of numeric features to be normalized
        cols_to_scale = ["ghi", "ghi_clear", "temp", "humidity", "wind",
            "clouds", "rain", "day_of_year", "month", "ghi_clear_ratio"]
        scaler = MinMaxScaler()
        self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])
        return self
    
    # Create csv documento with prepared dataset
    def save_csv(self, output_path):
        self.df.to_csv(output_path, index=False)
        return self

processor = DataProcessor("data/raw/data_set_raw.csv")

processor.load_rawdata()\
         .rename_features()\
         .format_date()\
         .remove_duplicates()\
         .generate_id()\
         .replace_missing()\
         .create_features()\
         .detect_outliers()\
         .remove_errors()\
         .encode_season()\
         .create_ghi_generation()\
         .create_label()\
         .normalize()\
         .save_csv("data/processed/data_set_prepared.csv")