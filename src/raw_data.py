import requests
import pandas as pd

class RawData:
    """
    Downloads daily solar and climate data from NASA POWER API
    for a given location and date range, and saves it as CSV.
    """

    def __init__(self, latitude, longitude, start_date, end_date, output_path):
        self.latitude = latitude
        self.longitude = longitude
        self.start_date = start_date  # format: YYYYMMDD
        self.end_date = end_date      # format: YYYYMMDD
        self.output_path = output_path
        self.parameters = [
            "ALLSKY_SFC_SW_DWN",    # GHI - Global Horizontal Irradiance
            "CLRSKY_SFC_SW_DWN",    # Clear-sky GHI
            "T2M",                  # Air temperature
            "RH2M",                 # Relative humidity
            "WS10M",                # Wind speed
            "CLOUD_AMT",            # Clouds
            "PRECTOTCORR"           # Total precipitation
        ]
        self.df = None

    def build_url(self):
        """
        Builds the NASA POWER API URL based on parameters and location.
        """

        param_str = ",".join(self.parameters)

        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"parameters={param_str}"
            "&community=RE"
            f"&longitude={self.longitude}"
            f"&latitude={self.latitude}"
            f"&start={self.start_date}"
            f"&end={self.end_date}"
            "&format=JSON"
        )

        
        return url
    
    def download(self):
        """
        Downloads data from NASA POWER and stores raw JSON.
        """

        url = self.build_url()
        print(f"Downloading NASA POWER data from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    
    def to_dataframe(self, data):
        """
        Converts NASA POWER JSON response into a pandas DataFrame.
        """

        records = data["properties"]["parameter"] # Extract the climate records

        df = pd.DataFrame() # Create the DataFrame
        for var in self.parameters: # Add each variable as a column
            df[var] = pd.Series(records[var])

        # Convert index (dates) into a column
        df = df.reset_index().rename(columns={"index": "date"})
        self.df = df
        return self

    def save_csv(self):
        """
        Saves the DataFrame to CSV at the specified output path.
        """

        if self.df is None:
            raise ValueError("DataFrame is empty. Call to_dataframe() first.")
        self.df.to_csv(self.output_path, index=False) # Save the final CSV
        print(f"CSV generated successfully at: {self.output_path}")
        return self

    def run(self):
        """
        Full pipeline: download → convert to DataFrame → save CSV.
        """

        data = self.download()
        self.to_dataframe(data)
        self.save_csv()
        return self


downloader = RawData(
    latitude=45.4215, # Coordinates of the location (example: Ottawa)
    longitude=-75.6972,
    start_date="20250701", # Period: jul/2025 to jun/2026
    end_date="20260630",
    output_path="data/raw/data_set_raw.csv"
)
downloader.run()