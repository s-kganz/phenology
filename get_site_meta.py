import earthaccess
import pandas as pd
import geopandas as gpd
import numpy as np
import statistics

earthaccess.login()

OUTPUT = "phenocam_site_meta.parquet"
OUTPUT_FILTERED = "phenocam_site_in_usa.parquet"
TARGET_VEG_TYPES = ["DB", "DN", "EB", "EN", "GR", "SH", "UN"]

# Get CONUS polygon for spatial filtering
CONUS = gpd.read_file("data_in/ne_110m_admin_1_states_provinces.shp")\
    .query("iso_a2 == 'US' and name != 'Alaska' and name != 'Hawaii'")\
    .union_all()

GRASS = pd.read_excel("data_in/nph12942-sup-0005-tables3.xls", skiprows=7)\
    .assign(genus = lambda x: x["Genus Authority"].str.split(" ", n=0).str[0].str.lower())

def read_site_meta(f):
    return pd.read_json(f)["phenocam_site"].transpose()

if __name__ == "__main__":
    # Get metadata, possibly from earthdata
    try:
        site_meta = pd.read_parquet(OUTPUT)
        print("Found cached site metadata")
    except FileNotFoundError:
        print("Cached metadata not found!")
        print("Searching for granules")
        meta_granules = earthaccess.search_data(
            short_name="PhenoCam_V3_2389",
            granule_name="*meta.json"
        )
        print(f"Found {len(meta_granules)} granules")
        site_meta = pd.DataFrame(map(read_site_meta, earthaccess.open(meta_granules))).set_index("sitename")
        site_meta.to_parquet(OUTPUT)

    # Filter on veg type
    site_meta_filter = site_meta[site_meta.primary_veg_type.isin(TARGET_VEG_TYPES)]
    print(f"After vegetation filter, {site_meta_filter.shape[0]} sites left")

    # Convert to geodataframe for spatial filter to CONUS
    site_meta_filter = gpd.GeoDataFrame(
        site_meta_filter,
        geometry=gpd.points_from_xy(site_meta_filter["lon"], site_meta_filter["lat"]),
        crs=4326
    ) 
    site_in_usa = site_meta_filter[site_meta_filter.within(CONUS)].copy()
    print(f"After CONUS filter, {site_in_usa.shape[0]} sites left")

    # Remap vegetation types
    site_in_usa["primary_veg_type"] = site_in_usa["primary_veg_type"].map({
        "DB": "Deciduous forest",
        "DN": "Deciduous forest",
        "GR": "Grassland",
        "EN": "Evergreen forest",
        "SH": "Shrubs",
        "EB": "Evergreen forest",
        "UN": "Forest understory"
    })

    # Classify sites as C3/C4 grass
    site_in_usa["genera"] = site_in_usa["dominant_species"]\
        .fillna("")\
        .str.lower()\
        .str.split(", ")\
        .apply(lambda x: [s.split(" ")[0] for s in x])
    
    grass_sites = site_in_usa[site_in_usa["primary_veg_type"] == "Grassland"].copy()
    grass_sites["grass_types"] = grass_sites["genera"].explode().map(
        GRASS.set_index("genus")["Photosynthetic types"]
    ).groupby(level=0).agg(lambda x: x.mode())

    # Manual fixes
    grass_sites.loc["kendall", "grass_types"] = 'C4'
    grass_sites.loc["sweetbriargrass", "grass_types"] = 'C4'
    grass_sites.loc["NEON.D03.DSNY.DP1.00042", "grass_types"] = 'C4'
    grass_sites.loc["NEON.D03.DSNY.DP1.00033", "grass_types"] = 'C4'
    grass_sites.loc["NEON.D11.OAES.DP1.00033", "grass_types"] = 'C4'
    grass_sites.loc["NEON.D11.OAES.DP1.00042", "grass_types"] = 'C4'
    grass_sites.loc["srg", "grass_types"] = 'C4'
    
    # Convert to human-readable veg type
    grass_sites["grass_types"][grass_sites["grass_types"].map(type) == np.ndarray] = np.nan
    grass_sites["grass_types"] = grass_sites["grass_types"].map({
        "C3": "C3 grass",
        "C4": "C4 grass"
    })

    site_in_usa.loc[grass_sites.index, "primary_veg_type"] = grass_sites["grass_types"]
    site_in_usa["primary_veg_type"].fillna("Unknown")

    # Diagnostic: site-years per veg type
    site_in_usa["site_years"] = (pd.to_datetime(site_in_usa["date_end"]) - pd.to_datetime(site_in_usa["date_start"])) / pd.Timedelta(days=365)

    print("Site-years per vegetation type")
    print(site_in_usa.groupby("primary_veg_type").agg({"site_years": "sum"})["site_years"].round())

    # Save results
    site_in_usa.to_parquet(OUTPUT_FILTERED)
