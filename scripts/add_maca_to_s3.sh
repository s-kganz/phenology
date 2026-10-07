#!/bin/bash

MACA_LINKS=(
    "http://thredds.northwestknowledge.net:8080/thredds/fileServer/NWCSC_INTEGRATED_SCENARIOS_ALL_CLIMATE/macav2livneh/BNU-ESM/macav2livneh_pr_BNU-ESM_r1i1p1_rcp85_2066_2085_CONUS_daily.nc"
    "http://thredds.northwestknowledge.net:8080/thredds/fileServer/NWCSC_INTEGRATED_SCENARIOS_ALL_CLIMATE/macav2livneh/BNU-ESM/macav2livneh_tasmax_BNU-ESM_r1i1p1_rcp85_2066_2085_CONUS_daily.nc" 
    "http://thredds.northwestknowledge.net:8080/thredds/fileServer/NWCSC_INTEGRATED_SCENARIOS_ALL_CLIMATE/macav2livneh/IPSL-CM5A-MR/macav2livneh_pr_IPSL-CM5A-MR_r1i1p1_rcp85_2066_2085_CONUS_daily.nc"
    "http://thredds.northwestknowledge.net:8080/thredds/fileServer/NWCSC_INTEGRATED_SCENARIOS_ALL_CLIMATE/macav2livneh/IPSL-CM5A-MR/macav2livneh_tasmax_IPSL-CM5A-MR_r1i1p1_rcp85_2066_2085_CONUS_daily.nc"
)

for item in "${MACA_LINKS[@]}"; do
    echo "$item"
    # Download the file
    wget -nc -c -nd "$item" -O "/tmp/$(basename $item)"
    
    # Copy to S3
    aws s3 cp "/tmp/$(basename $item)" "s3://nasa-cryo-persistent/ganzk/maca/$(basename $item)"

    # Delete file from /tmp/
    rm "/tmp/$(basename $item)"
done