import os
import pandas as pd
import rasterio
from tqdm import tqdm
from datetime import datetime

station_csv = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\Integration\Station\stations_obs_2019_2022.csv"
slowvars_dir = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\GEE\slow"
fastvars_dir = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\GEE\fast"
rainvars_dir = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\comephore\aligned"
output_csv   = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\Integration\allValue\design_matrix_hourly.csv"


def sample_raster(raster_path, lon, lat, band=None):
    """EPSG:4326band=None"""
    with rasterio.open(raster_path) as src:
        row, col = src.index(lon, lat)
        if not (0 <= row < src.height and 0 <= col < src.width):
            return None if band else [None] * src.count
        if band:
            return src.read(band, window=((row, row+1), (col, col+1)))[0,0]
        else:
            return [src.read(i+1, window=((row, row+1), (col, col+1)))[0,0] for i in range(src.count)]

def find_band_index(hour, var_type):
    """

    fastvars:
        t2m: band 1–13
        swvl1: band 14–26
    rainvars:
        rain: band 1–13
    """
    hour_index = hour - 6  # 06:00 is 0
    if var_type == "t2m":
        return hour_index + 1
    elif var_type == "swvl1":
        return hour_index + 14
    elif var_type == "rain":
        return hour_index + 1
    else:
        raise ValueError("Unknown var_type")


df_obs = pd.read_csv(station_csv, parse_dates=["time_local"])

records = []


for _, row in tqdm(df_obs.iterrows(), total=len(df_obs), desc="Processing observations"):
    st_id, lon, lat, temp_obs, dt_local = row["station_id"], row["lon"], row["lat"], row["Temperature"], row["time_local"]
    year, month, day, hour = dt_local.year, dt_local.month, dt_local.day, dt_local.hour

    slow_file = f"slowvars_{year}{month:02d}_modis1km.tif"
    fast_file = f"fastvars_{year}{month:02d}{day:02d}_local0618_modis1km.tif"
    rain_file = f"rainvars_{year}{month:02d}{day:02d}_local0618_modis1km.tif"

    slow_path = os.path.join(slowvars_dir, slow_file)
    fast_path = os.path.join(fastvars_dir, fast_file)
    rain_path = os.path.join(rainvars_dir, rain_file)

    # slow
    slow_vals = sample_raster(slow_path, lon, lat) if os.path.exists(slow_path) else [None]*4
    emis31, ndvi, mndwi, elev = slow_vals

    # =fast
    t2m_val = sample_raster(fast_path, lon, lat, band=find_band_index(hour, "t2m")) if os.path.exists(fast_path) else None
    swvl1_val = sample_raster(fast_path, lon, lat, band=find_band_index(hour, "swvl1")) if os.path.exists(fast_path) else None

    # rain
    rain_val = sample_raster(rain_path, lon, lat, band=find_band_index(hour, "rain")) if os.path.exists(rain_path) else None

    records.append({
        "station_id": st_id,
        "lon": lon,
        "lat": lat,
        "time_local": dt_local,
        "Temperature": temp_obs,
        "emis31": emis31,
        "ndvi": ndvi,
        "mndwi": mndwi,
        "elevation": elev,
        "t2m": t2m_val,
        "swvl1": swvl1_val,
        "rain": rain_val
    })

# ==== 保存 ====
df_out = pd.DataFrame(records)
df_out.to_csv(output_csv, index=False)
print(f"success: {output_csv}")
