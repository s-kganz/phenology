import requests
import pandas as pd
from src.daymet import extract_timeseries
from src.ameriflux import amf_download_base, amf_site_timeseries
from tqdm import tqdm
import zipfile
import os
import numpy as np

SITE_META = "phenocam_site_in_usa.parquet"
DAYMET_VARS = ["tmin", "tmax", "prcp", "vp"]
OUTPUT = "phenocam_site_in_usa_climate.parquet"
START_YEAR = 1990
END_YEAR = 2025

def get_site_timeseries(sitename: str, rowmeta: pd.Series) -> pd.DataFrame:
    # Read climatology from Daymet by default. Overwrite with Ameriflux
    # if available.
    lat = rowmeta["lat"]
    lon = rowmeta["lon"]
    site_ts = extract_timeseries(START_YEAR, END_YEAR, lat, lon, DAYMET_VARS)
    site_ts["src"] = "daymet"

    if (
        rowmeta["flux_data"] and 
        "AMERIFLUX" in rowmeta["flux_networks"] and
        (rowmeta["flux_sitenames"] != np.nan)
    ):
        print(f"Trying flux data for {sitename}")
        try:
            site_base = amf_download_base(rowmeta["flux_sitenames"], "CCBY4.0", "remote_sensing", "phenology model")[0]
            with zipfile.ZipFile(site_base, 'r') as z:
                base_file = next(filter(lambda x: "BASE" in x and x.endswith(".csv"), z.namelist()))
                site_base_df = pd.read_csv(z.open(base_file), comment="#")
                site_ts_amf = amf_site_timeseries(site_base_df)
                site_ts_amf["src"] = "ameriflux"
                site_ts.update(site_ts_amf)
        except RuntimeError as e:
            if "Cannot find data" in str(e):
                print(f"Failed to find data for {sitename} using AMF sitename {rowmeta["flux_sitenames"]}")
            else:
                raise

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
