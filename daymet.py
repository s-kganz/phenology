import requests
import io
import pandas as pd

def extract_timeseries(start_year: int, end_year: int, lat: float, lon: float, vars: list[str]) -> pd.DataFrame:
    years_comma_sep = ",".join(map(str, range(start_year, end_year+1))) # inclusive
    vars_comma_sep  = ",".join(vars)
    request_url = f"https://daymet.ornl.gov/single-pixel/api/data?lat={lat}&lon={lon}&vars={vars_comma_sep}&years={years_comma_sep}"
    
    resp = requests.get(request_url)
    
    df = pd.read_csv(io.StringIO(resp.text), skiprows=6)
    
    return df