import requests
import pandas as pd
import itertools
from tqdm import tqdm
import xarray as xr

ESM = ["BNU-ESM", "IPSL-CM5A-MR"]
VAR = ["pr", "tasmax"]

SITE_META = "phenocam_site_in_usa.parquet"
OUTPUT = "phenocam_site_in_usa_climate_maca.parquet"

def get_site_timeseries(sitename: str, rowmeta: pd.Series, dataset) -> pd.DataFrame:
    lat = rowmeta["lat"]
    lon = rowmeta["lon"]
    site_ts = extract_timeseries(START_YEAR, END_YEAR, lat, lon, DAYMET_VARS)
    site_ts["site"] = sitename
    return site_ts

if __name__ == "__main__":
    meta = pd.read_parquet(SITE_META)

    lat_da = xr.DataArray(meta["lat"], dims="site", coords=dict(site=meta.index.values))
    # Force positive longitude values
    lon_da = xr.DataArray(meta["lon"] % 360, dims="site", coords=dict(site=meta.index.values))

    dfs_by_site = []

    for (this_esm, this_var) in itertools.product(ESM, VAR):
        # Open dataset
        s3_link = f"s3://nasa-cryo-persistent/ganzk/maca/macav2livneh_{this_var}_{this_esm}_r1i1p1_rcp85_2066_2085_CONUS_daily.nc"
        ds = xr.open_dataset(s3_link, engine="h5netcdf") # h5netcdf necessary for s3

        # Select TS at points of interest
        ds_ts = ds.sel(lat=lat_da, lon=lon_da, method="nearest")
        ds_ts = ds_ts.compute().drop_vars(["crs", "lat", "lon"])

        # Convert to pandas
        df = ds_ts.to_dataframe()
        df.columns = [f"{this_esm}_{this_var}"]
        dfs_by_site.append(df)

    pd.concat(dfs_by_site, axis=1, join="outer").to_parquet(OUTPUT)

