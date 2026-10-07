import requests
import pandas as pd
from src.daymet import extract_timeseries
from tqdm import tqdm

SITE_META = "phenocam_site_in_usa.parquet"
DAYMET_VARS = ["tmin", "tmax", "prcp", "vp"]
OUTPUT = "phenocam_site_in_usa_climate.parquet"
START_YEAR = 1990
END_YEAR = 2025

def get_site_timeseries(sitename: str, rowmeta: pd.Series) -> pd.DataFrame:
    lat = rowmeta["lat"]
    lon = rowmeta["lon"]
    site_ts = extract_timeseries(START_YEAR, END_YEAR, lat, lon, DAYMET_VARS)
    site_ts["site"] = sitename
    return site_ts

if __name__ == "__main__":
    # Iterate over each site, send request, and assemble results
    # back into a dataframe.
    meta = pd.read_parquet(SITE_META)

    all_site_ts = []

    with tqdm(meta.iterrows(), total=meta.shape[0]) as pbar:
        for i, r in pbar:
            pbar.set_description(i)
            all_site_ts.append(get_site_timeseries(i, r))

    all_site_ts_df = pd.concat(all_site_ts)

    all_site_ts_df.to_parquet(OUTPUT)
