import requests
import pandas as pd

# Coordinates of the location (example: Ottawa)
latitude = 45.4215
longitude = -75.6972

# Period: jul/2025 to jun/2026
start_date = "20250701"
end_date   = "20260630"

# Essential climate variables for solar energy production
parameters = [
    "ALLSKY_SFC_SW_DWN",   # Global solar irradiance (GHI)
    "CLRSKY_SFC_SW_DWN",   # Clear-sky irradiance
    "T2M",                 # Air temperature
    "RH2M",                # Relative humidity
    "WS10M",               # Wind speed
    "CLOUD_AMT",           # Cloud amount
    "PRECTOTCORR"          # Total precipitation
]

# Build the URL
param_str = ",".join(parameters)

url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point?"
    f"parameters={param_str}"
    "&community=RE"
    f"&longitude={longitude}"
    f"&latitude={latitude}"
    f"&start={start_date}"
    f"&end={end_date}"
    "&format=JSON"
)

print("Downloading NASA POWER data...")
response = requests.get(url)
data = response.json()

# Extract the climate records
records = data["properties"]["parameter"]

# Create the DataFrame
df = pd.DataFrame()

# Add each variable as a column
for var in parameters:
    df[var] = pd.Series(records[var])

# Convert the index (dates) into a column
df = df.reset_index().rename(columns={"index": "date"})

# Save the final CSV
df.to_csv("data/raw/nasa_power_solar_raw_2026.csv", index=False)

print("CSV generated successfully!")
