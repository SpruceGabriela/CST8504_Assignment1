import requests
import pandas as pd

# Coordinates of the location (example: Ottawa)
latitude = 45.4215
longitude = -75.6972

# Period: April to September 2026
start_date = "20260401"
end_date   = "20260930"

# Climate variables relevant for solar energy production
parameters = [
    "ALLSKY_SFC_SW_DWN",   # Solar irradiance (GHI) - Irradiancia solar (GHI)
    "CLRSKY_SFC_SW_DWN",   # Clear-sky irradiance   - radiacao infravermelha
    "ALLSKY_SFC_LW_DWN",   # Longwave radiation     - irradiancia em ceu limpo
    "T2M",                 # Air temperature        - temperatura
    "RH2M",                # Relative humidity      - umidade
    "WS10M",               # Wind speed             - vento
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

# Create the DataFrame correctly (dates as rows, variables as columns)
df = pd.DataFrame()

# Add each variable as a column
for var in parameters:
    df[var] = pd.Series(records[var])

# Convert the index (dates) into a column
df = df.reset_index().rename(columns={"index": "date"})

# Save the final CSV
df.to_csv("nasa_power_apr_sep_2026.csv", index=False)

print("CSV generated successfully!")
