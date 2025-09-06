
import os
import re
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from datetime import datetime
import pytz

SLOW_REF_FILE = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\GEE\slow\201906.tif"
COMEPHORE_BASE = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\comephore\downloadData"
OUTPUT_DIR = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\comephore\updateDir"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# time
YEARS = {2019, 2022}          # year
MONTHS = {6, 7, 8}            # 6-8
LOCAL_HOURS_END = list(range(6, 19))  # 本local hour

# time zone
tz_paris = pytz.timezone("Europe/Paris")
tz_utc = pytz.utc

# file name
rr_pattern = re.compile(r"(\d{4})(\d{2})(\d{2})(\d{2})_RR\.g?tif$", re.IGNORECASE)

#read slow grid
with rasterio.open(SLOW_REF_FILE) as ref:
    ref_meta = ref.meta.copy()
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_width = ref.width
    ref_height = ref.height
    print("[REF] CRS:", ref_crs)
    print("[REF] res:", ref.res)
    print("[REF] size:", ref_width, "x", ref_height)

def parse_utc_end_from_name(y, m, d, hh):
    #end time
    return tz_utc.localize(datetime(y, m, d, hh))

for folder in sorted(os.listdir(COMEPHORE_BASE)):
    folder_path = os.path.join(COMEPHORE_BASE, folder)
    if not os.path.isdir(folder_path):
        continue

    daily = {}  # key: local_date_str, val: dict{local_hour_end: (array, band_desc)}

    for fn in sorted(os.listdir(folder_path)):
        m = rr_pattern.match(fn)
        if not m:
            continue
        y, mth, d, hh = map(int, m.groups())

        dt_utc_end = parse_utc_end_from_name(y, mth, d, hh)
        dt_loc_end = dt_utc_end.astimezone(tz_paris)

        if YEARS and dt_loc_end.year not in YEARS:
            continue
        if dt_loc_end.month not in MONTHS:
            continue

        # 06..18
        if dt_loc_end.hour not in LOCAL_HOURS_END:
            continue

        # 1/10 mm,nan65535）
        in_path = os.path.join(folder_path, fn)
        with rasterio.open(in_path) as src:
            data = src.read(1).astype(np.float32)
            out_arr = np.full((ref_height, ref_width), -9999.0, dtype=np.float32)
            reproject(
                source=data,
                destination=out_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
                src_nodata=65535,
                dst_nodata=-9999.0,
            )

            valid = out_arr != -9999.0
            out_arr[valid] = out_arr[valid] / 10.0

        local_date_str = dt_loc_end.strftime("%Y%m%d")
        band_name = f"rain_{local_date_str}T{dt_loc_end.strftime('%H')}"
        daily.setdefault(local_date_str, {})[dt_loc_end.hour] = (out_arr, band_name)

    for local_date_str, hour_map in daily.items():
        arrays, descs = [], []
        for hh in LOCAL_HOURS_END:  # 06..18
            if hh in hour_map:
                arr, name = hour_map[hh]
            else:
                arr = np.full((ref_height, ref_width), -9999.0, dtype=np.float32)
                name = f"rain_{local_date_str}T{hh:02d}"
            arrays.append(arr)
            descs.append(name)

        out_path = os.path.join(
            OUTPUT_DIR,
            f"rainvars_{local_date_str}_local06to18_slow.tif"
        )
        meta = ref_meta.copy()
        meta.update({
            "count": len(arrays),      # 13
            "dtype": "float32",
            "nodata": -9999.0,
            "compress": "lzw"
        })
        with rasterio.open(out_path, "w", **meta) as dst:
            for i, (arr, name) in enumerate(zip(arrays, descs), start=1):
                dst.write(arr, i)

                dst.set_band_description(i, name)

        print(f"✅ {local_date_str}: {len(arrays)} bands → {out_path}")
