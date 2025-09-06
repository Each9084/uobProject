
import math
import os
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box, mapping

# path
SHAPE_DIR = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\GDAM\GadmOutput"
SHAPE_NAME = "paris_core.shp"

#
REL_BUFFER = 0.15      # expand 15%
ABS_BUFFER_KM = 15.0
ABS_BUFFER_DEG = 0.10  # expand 0.10°
SNAP_DEG = 0.05        # merge to  0.05° grid

#
def km_to_deg_lat(km: float) -> float:

    return km / 111.32

def km_to_deg_lon(km: float, lat_deg: float) -> float:
    # 111.32 * cos(lat) km
    return km / (111.32 * max(math.cos(math.radians(lat_deg)), 1e-6))

def snap(x: float, step: float, mode="out"):

    if mode == "out":
        if x >= 0:
            return math.floor(x / step) * step
        else:
            return math.ceil(x / step) * step
    else:
        return round(x / step) * step

#
shape_path = Path(SHAPE_DIR) / SHAPE_NAME
if not shape_path.exists():
    raise FileNotFoundError(f"unfound：{shape_path}")

gdf = gpd.read_file(shape_path)

#
if gdf.crs is None:

    raise ValueError("CRS NULL")
if gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

minx, miny, maxx, maxy = gdf.total_bounds
width_deg = maxx - minx
height_deg = maxy - miny
lat0 = (miny + maxy) / 2.0

print("Original：")
print(f"  lon_min={minx:.6f}, lon_max={maxx:.6f}, lat_min={miny:.6f}, lat_max={maxy:.6f}")
print(f"  width_deg={width_deg:.6f}, height_deg={height_deg:.6f} (degree)")



rel_buf_lon_deg = width_deg * REL_BUFFER
rel_buf_lat_deg = height_deg * REL_BUFFER


abs_lat_deg_from_km = km_to_deg_lat(ABS_BUFFER_KM)
abs_lon_deg_from_km = km_to_deg_lon(ABS_BUFFER_KM, lat0)

buf_lat_deg = max(rel_buf_lat_deg, ABS_BUFFER_DEG, abs_lat_deg_from_km)
buf_lon_deg = max(rel_buf_lon_deg, ABS_BUFFER_DEG, abs_lon_deg_from_km)


raw_minx = minx - buf_lon_deg
raw_maxx = maxx + buf_lon_deg
raw_miny = miny - buf_lat_deg
raw_maxy = maxy + buf_lat_deg


snap_minx = snap(raw_minx, SNAP_DEG, mode="out")
snap_maxx = snap(raw_maxx + (SNAP_DEG - 1e-12), SNAP_DEG, mode="out")
snap_miny = snap(raw_miny, SNAP_DEG, mode="out")
snap_maxy = snap(raw_maxy + (SNAP_DEG - 1e-12), SNAP_DEG, mode="out")


snap_minx = max(-180.0, snap_minx)
snap_maxx = min(180.0, snap_maxx)
snap_miny = max(-90.0, snap_miny)
snap_maxy = min(90.0, snap_maxy)


extent_poly = box(snap_minx, snap_miny, snap_maxx, snap_maxy)

print(f"  lon_min={snap_minx:.4f}, lon_max={snap_maxx:.4f}, lat_min={snap_miny:.4f}, lat_max={snap_maxy:.4f}")


def deg_width_km(lon_min, lon_max, lat_deg):
    return (lon_max - lon_min) * 111.32 * math.cos(math.radians(lat_deg))
def deg_height_km(lat_min, lat_max):
    return (lat_max - lat_min) * 111.32

center_lat = (snap_miny + snap_maxy) / 2.0
width_km = deg_width_km(snap_minx, snap_maxx, center_lat)
height_km = deg_height_km(snap_miny, snap_maxy)
print(f"  approx_width_km={width_km:.1f}, approx_height_km={height_km:.1f}")


out_geojson = Path(SHAPE_DIR) / "paris_research_extent.geojson"
with open(out_geojson, "w", encoding="utf-8") as f:
    json.dump({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "name": "paris_research_extent",
                "snap_deg": SNAP_DEG,
                "rel_buffer": REL_BUFFER,
                "abs_buffer_km": ABS_BUFFER_KM,
                "abs_buffer_deg": ABS_BUFFER_DEG
            },
            "geometry": mapping(extent_poly)
        }]
    }, f, ensure_ascii=False)

print(f"\n output GeoJSON：{out_geojson}")


try:
    gpd.GeoDataFrame({"name": ["paris_research_extent"]},
                     geometry=[extent_poly], crs="EPSG:4326") \
      .to_file(Path(SHAPE_DIR) / "paris_research_extent.shp")
    print("Shapefile：", Path(SHAPE_DIR) / "paris_research_extent.shp")
except Exception as e:
    print("Shapefile ：", e)


