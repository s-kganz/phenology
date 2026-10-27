import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu

META = pd.read_parquet("phenocam_site_in_usa.parquet")
PHENO = pd.read_parquet("phenocam_site_in_usa_phenology.parquet")
GCC_FIELD = "gcc_90"
OUTPUT = "phenocam_phenology_model.parquet"


if __name__ == "__main__":
    # Select greenness field
    pheno_clean = PHENO.query(f"gcc_value == '{GCC_FIELD}'")

    # Convert date columns to date dtype
    date_columns = list(filter(lambda x: x.startswith("transition_"), pheno_clean.columns))
    pheno_clean[date_columns] = pheno_clean[date_columns].apply(pd.to_datetime)

    # Determine calendar year of each transition
    pheno_clean = pheno_clean.assign(
        transition_year = lambda x: x.transition_50.dt.year
    )

    # Determine which sites have a multimodal transition distribution
    greenups_per_year = (
        pheno_clean
        .query("direction == 'rising'")
        .groupby(["site", "transition_year", "veg_type", "roi_id"])
        ["transition_50"]
        .size()
    )

    prop_multi_cycle = (
        greenups_per_year
        .groupby("site")
        .agg(lambda x: (x > 1).mean())
    )

    multi_modal_sites = prop_multi_cycle[prop_multi_cycle > 0].index

    pheno_clean["is_multimodal"] = pheno_clean["site"].isin(multi_modal_sites)

    # Keep multimodal rises in the first peak and multimodal falls in the second peak
    rise_threshold = threshold_otsu(
        pheno_clean[
            (pheno_clean.direction == "rising") &
            (pheno_clean.is_multimodal)
        ].transition_50.dt.dayofyear.to_numpy()
    )

    fall_threshold = threshold_otsu(
        pheno_clean[
            (pheno_clean.direction == "falling") &
            (pheno_clean.is_multimodal)
        ].transition_50.dt.dayofyear.to_numpy()
    )

    print(f"# multimodal sites: {multi_modal_sites.size}")
    print(f"Rising threshold DOY: {rise_threshold}")
    print(f"Falling threshold DOY: {fall_threshold}")

    pheno_clean = pheno_clean[
        (~pheno_clean.is_multimodal) |
        ((pheno_clean.direction == "rising") & (pheno_clean.transition_50.dt.dayofyear <= rise_threshold)) |
        ((pheno_clean.direction == "falling") & (pheno_clean.transition_50.dt.dayofyear >= fall_threshold))
    ]

    # Get DOY of transition dates
    yday_columns = list(map(lambda x: "doy_" + x, date_columns))

    for col in date_columns:
        new_col = "doy_"+col
        pheno_clean[new_col] = pheno_clean[col].dt.dayofyear

    # Calculate slopes
    pheno_clean = pheno_clean.assign(
        # Relativize greenenss transitions for slope calculation
        rel_green_50=lambda x: np.minimum(1, (x.threshold_50 - x.min_gcc) / (x.max_gcc - x.min_gcc)),
        rel_green_10=lambda x: np.maximum(0, (x.threshold_10 - x.min_gcc) / (x.max_gcc - x.min_gcc)),
    )

    pheno_clean = pheno_clean.assign(
        transition_slope = lambda x: np.abs((np.log(2 * x.rel_green_50 / x.rel_green_10) - 1) / (x.doy_transition_50 - x.doy_transition_10))
    )

    # Save output
    pheno_clean.to_parquet(OUTPUT)