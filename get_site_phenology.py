import pandas as pd
import earthaccess
from tqdm import tqdm
import io
import os

earthaccess.login()

SITE_META = "phenocam_site_in_usa.parquet"
OUTPUT = "phenocam_site_in_usa_phenology.parquet"

def read_site_phenology(f: earthaccess.store.EarthAccessFile) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(f.read()), skiprows=16)

if __name__ == "__main__":
    meta = pd.read_parquet(SITE_META)

    # Get all granules for pheno transitions
    transition_granules = earthaccess.search_data(
        short_name="PhenoCam_V3_2389",
        granule_name="*1day_transition_dates*"
    )
    print(f"Found {len(transition_granules)} granules")

    # Only keep granules that are in our sitelist
    transition_granules_filter = list(filter(
        lambda x: os.path.basename(x.data_links()[0]).split("_")[0] in meta.index,
        transition_granules
    ))
    print(f"After filtering, {len(transition_granules_filter)} remaining")

    # Open all the files
    transition_file_objs = earthaccess.open(transition_granules_filter)

    # Read them in as dataframes
    transition_dfs = tqdm(map(read_site_phenology, transition_file_objs), total=len(transition_file_objs))

    # Concat and save
    pd.concat(transition_dfs).to_parquet(OUTPUT)
    
