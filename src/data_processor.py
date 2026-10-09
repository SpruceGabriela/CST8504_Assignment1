import pandas as pd
import numpy as np


# Load dataset
df = pd.read_csv('data/raw/nasa_power_solar_raw_2026.csv')

# Date format
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

# Remove duplicates
df = df.drop_duplicates()

# Generate ID
df["id"] = range(1, len(df) + 1)

# Move ID column as 1st column of dataset
df = df[["id"] + [col for col in df.columns if col != "id"]]

# Replace NASA_POWER missing values (-999.0) with NaN
df = df.replace(-999.0, np.nan)

# Outliers detection (IQR method)
def detect_outliers(series):
    Q1 = series.quantile(0.25) # values on 1st quarter
    Q3 = series.quantile(0.75) # values on 3rd quarter
    IQR = Q3 - Q1 # range of the middle half of the dataset
    lower = Q1 - 1.5 * IQR # lower limit
    upper = Q3 + 1.5 * IQR # upper limit
    return (series < lower) | (series > upper)

numeric_cols = df.select_dtypes(include=[np.number]).columns # select columns with numeric data types and pick the columns names to use on next line
outlier_flags = df[numeric_cols].apply(detect_outliers) # apply the function detect_outliers on numeric columns

df_outliers = df[outlier_flags.any(axis=1)].copy() # verify every line, and select/copy instance where the outliers_flag is true

# Outliers classification
def classify_outlier(row):
    # Real events, not outliers
    if row["PRECTOTCORR"] > 15:
        return "heavy rain (real)"
    if row["CLOUD_AMT"] > 95:
        return "dense clouds (real)"
    if row["T2M"] < -25:
        return "extreme cold (real)"
    if row["T2M"] > 35:
        return "extreme hot (real)"

    # Possible sensor errors
    if row["ALLSKY_SFC_SW_DWN"] < 0.5 and row["CLOUD_AMT"] < 20:
        return "possible sensor error"

    return "normal meteorological outlier"

df_outliers["classification"] = df_outliers.apply(classify_outlier, axis=1)

# REMOVE ONLY OUTLIERS CLASSIFIED AS SENSOR ERROR
errors = df_outliers[df_outliers["classification"] == "possible sensor error"]
df_clean = df[~df.index.isin(errors.index)].copy()

# Interpolate missing values (best for time series)
df = df.interpolate(method="linear")

# Create features day_of_year, month, season
df["day_of_year"] = df["date"].dt.dayofyear
df["month"] = df["date"].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"
df["season"] = df["month"].apply(get_season)

# Create feature ghi_clear_ratio = real irradiance / clear-sky irradiance
df["ghi_clear_ratio"] = df["ALLSKY_SFC_SW_DWN"] / df["CLRSKY_SFC_SW_DWN"]

# Analising the dataset
# print(df.head())       # First datas
# print(df.describe())   # statistics
# print(df_outliers[["date", "classification"]])
# errors[["date", "classification"]]


# Create csv documento with prepared dataset
df.to_csv("data/processed/nasa_power_prepared.csv", index=False)
