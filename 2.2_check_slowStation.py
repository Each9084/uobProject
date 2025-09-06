from pathlib import Path
import pandas as pd
import numpy as np
import rasterio as rio
from rasterio.sample import sample_gen
from rasterio.transform import rowcol
from pyproj import Transformer


SLOW_DIR = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\GEE\slow"

RASTERS = [
    "Elevation_WGS84_24x17.tif",
    "Emis31_2019_2019-06_WGS84_24x17.tif",
    "NDVI_2019_2019-06_WGS84_24x17.tif",
    "MNDWI_2019_2019-06_WGS84_24x17.tif",
]


OUT_CSV = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\GEE\slow\stations_with_slow_vars.csv"

stations_data = [
    {"station_id": "Station A", "lat": 48.84469985961914, "lon": 2.3338000774383545},
    {"station_id": "Station B", "lat": 48.82170104980469, "lon": 2.3378000259399414},
    {"station_id": "Station C", "lat": 48.82170104980469, "lon": 2.3378000259399414},
    {"station_id": "Station D", "lat": 48.85480117797852, "lon": 2.2337000370025635},
{"station_id": "000CT Paris 6ème", "lat": 48.85, "lon": 2.24},
{"station_id": "000EW Paris 20ème", "lat": 48.85, "lon": 2.41},
]


def load_rasters(base_dir, raster_names):

    opened = {}
    for name in raster_names:
        fp = Path(base_dir) / name
        if not fp.exists():
            raise FileNotFoundError(f"栅格不存在: {fp}")
        ds = rio.open(fp)
        opened[name] = ds
    return opened

def build_transformer(dst_crs):
    """(EPSG:4326) -> CRS """
    return Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)

def sample_one(ds, x_dst, y_dst):
    try:
        # sample [(x,y)]
        val = list(ds.sample([(x_dst, y_dst)]))[0][0]
        # nodata
        nodata = ds.nodatavals[0]
        if nodata is not None and val == nodata:
            return np.nan
        return float(val) if np.isfinite(val) else np.nan
    except Exception:
        return np.nan

def point_in_bounds(ds, x_dst, y_dst):
    left, bottom, right, top = ds.bounds
    return (left <= x_dst <= right) and (bottom <= y_dst <= top)

def rowcol_of_point(ds, x_dst, y_dst):
    try:
        r, c = rowcol(ds.transform, x_dst, y_dst)
        return int(r), int(c)
    except Exception:
        return None, None

def main():
    # station table
    stations_df = pd.DataFrame(stations_data)
    stations_df = stations_df.copy()
    stations_df["lon"] = stations_df["lon"].astype(float)
    stations_df["lat"] = stations_df["lat"].astype(float)

    # open tif
    rasters = load_rasters(SLOW_DIR, RASTERS)

    # switch crs
    template = list(rasters.values())[0]
    dst_crs = template.crs
    transformer = build_transformer(dst_crs)
    out_rows = []
    for _, row in stations_df.iterrows():
        sid = row["station_id"]
        lon = float(row["lon"])
        lat = float(row["lat"])
        x_dst, y_dst = transformer.transform(lon, lat)
        in_bounds = point_in_bounds(template, x_dst, y_dst)
        rr, cc = rowcol_of_point(template, x_dst, y_dst)

        record = {
            "station_id": sid,
            "lon": lon,
            "lat": lat,
            "proj_x": x_dst,
            "proj_y": y_dst,
            "row": rr,
            "col": cc,
            "out_of_bounds": not in_bounds,
        }

        for name, ds in rasters.items():
            val = np.nan
            if point_in_bounds(ds, x_dst, y_dst):
                val = sample_one(ds, x_dst, y_dst)

            key = Path(name).stem
            record[key] = val

        out_rows.append(record)

    result = pd.DataFrame(out_rows)
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] ：{OUT_CSV}")
    cols_preview = ["station_id", "lat", "lon", "row", "col", "out_of_bounds"] + [Path(n).stem for n in RASTERS]
    print(result[cols_preview])

    for ds in rasters.values():
        ds.close()

if __name__ == "__main__":
    main()
