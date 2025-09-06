import os, glob, math, warnings, re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import box, Polygon
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer, Geod
import matplotlib.pyplot as plt



DATA_DIR = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\GEE\slow"

# GEE  ROI（WGS84）
ROI_WGS84 = box(2.15, 48.75, 2.55, 48.95)
STATS_ON_ROI = True
MAKE_QUICKLOOK = True

# WGS84 EW
ONE_KM_TOL = 0.15  # 15%
ALIGN_TOL_PIX = 0.51
CSV_OUT = os.path.join(DATA_DIR, "slowvars_check_summary.csv")


@dataclass
class RasterInfo:
    path: str
    name: str
    year: Optional[int]
    month: Optional[int]
    crs: Optional[CRS]
    width: int
    height: int
    transform: Affine
    bounds: Tuple[float, float, float, float]
    pixel_size_x: float
    pixel_size_y: float
    px_m_east_west: float
    px_m_north_south: float
    approx_1km_ok: bool
    band_count: int
    band_descriptions: Tuple
    dtypes: List[str]
    nodata: List[Optional[float]]
    stats: List[Tuple[float, float, float]]  # (min,max,mean)
    overlap_ratio_roi: float  # ROI
    overlap_ratio_img: float
    align_ok: bool



def geodesic_pixel_meters(lon, lat, dx_deg, dy_deg) -> Tuple[float, float]:
    geod = Geod(ellps="WGS84")
    _, _, ew = geod.inv(lon, lat, lon + dx_deg, lat)
    _, _, ns = geod.inv(lon, lat, lon, lat + dy_deg)
    return abs(ew), abs(ns)

def polygon_geodesic_area(poly_wgs84: Polygon) -> float:
    geod = Geod(ellps="WGS84")
    x, y = poly_wgs84.exterior.coords.xy
    area, _ = geod.polygon_area_perimeter(x, y)
    return abs(area)

def reproject_polygon(poly: Polygon, src_crs: CRS, dst_crs: CRS) -> Polygon:
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_transform(lambda x, y: transformer.transform(x, y), poly)

def compute_overlap_ratios(img_poly_wgs84: Polygon, roi_wgs84: Polygon) -> Tuple[float, float]:
    inter = img_poly_wgs84.intersection(roi_wgs84)
    if inter.is_empty:
        return 0.0, 0.0
    area_roi = polygon_geodesic_area(roi_wgs84)
    area_img = polygon_geodesic_area(img_poly_wgs84)
    area_inter = polygon_geodesic_area(inter)
    return (area_inter / area_roi, area_inter / area_img)

def check_alignment_to_roi_grid(ds: rasterio.DatasetReader, roi_wgs84: Polygon) -> bool:
    if not (ds.crs and ds.crs.to_epsg() == 4326):
        return False
    left, bottom, right, top = ds.bounds
    xres = ds.transform.a
    yres = -ds.transform.e
    def near_integer(x, tol=ALIGN_TOL_PIX):
        return abs(x - round(x)) <= tol
    dx_left   = (roi_wgs84.bounds[0] - left)   / xres
    dx_right  = (roi_wgs84.bounds[2] - left)   / xres
    dy_bottom = (roi_wgs84.bounds[1] - bottom) / yres
    dy_top    = (roi_wgs84.bounds[3] - bottom) / yres
    return all(near_integer(v) for v in [dx_left, dx_right, dy_bottom, dy_top])

def approx_1km_lataware(crs, bounds, dx_deg, dy_deg, px_m_ew, px_m_ns, tol=ONE_KM_TOL):
    if crs and CRS.from_user_input(crs).to_epsg() == 4326:
        lat_c = (bounds[1] + bounds[3]) / 2.0
        expect_ns = 1000.0
        expect_ew = 1000.0 * math.cos(math.radians(lat_c))
        ok_ns = abs(px_m_ns - expect_ns) <= expect_ns * tol
        ok_ew = abs(px_m_ew - expect_ew) <= max(25.0, expect_ew * tol)
        return bool(ok_ns and ok_ew)
    else:
        if not (math.isnan(px_m_ew) or math.isnan(px_m_ns)):
            return (800.0 <= px_m_ew <= 1200.0) and (800.0 <= px_m_ns <= 1200.0)
        return False

def read_stats_nan_safe(arr: np.ndarray) -> Tuple[float, float, float]:
    a = arr.astype("float64", copy=False)
    a[~np.isfinite(a)] = np.nan
    return float(np.nanmin(a)), float(np.nanmax(a)), float(np.nanmean(a))

def stats_on_roi(ds: rasterio.DatasetReader, roi_wgs84: Polygon) -> List[Tuple[float, float, float]]:
    if ds.crs and ds.crs.to_epsg() != 4326:
        trans = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        roi_img = shp_transform(lambda x, y: trans.transform(x, y), roi_wgs84)
    else:
        roi_img = roi_wgs84
    rb = roi_img.bounds
    win = window_from_bounds(rb[0], rb[1], rb[2], rb[3], ds.transform)
    stats = []
    for b in range(1, ds.count + 1):
        data = ds.read(b, window=win, masked=False)
        stats.append(read_stats_nan_safe(data))
    return stats

def parse_year_month_from_name(name: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r'(20\d{2})(0[1-9]|1[0-2])', name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def make_quicklook(path: str, out_dir: str, title: str):
    try:
        with rasterio.open(path) as ds:
            use_band = 1
            descs = ds.descriptions or ()
            for i, d in enumerate(descs, start=1):
                if d and any(k in d.lower() for k in ("ndvi", "mndwi", "emis", "elevation")):
                    use_band = i
                    break
            data = ds.read(use_band, masked=False).astype("float64")
            data[~np.isfinite(data)] = np.nan
            good = data[np.isfinite(data)]
            if good.size == 0:
                raise ValueError("All values are NaN.")
            vmin = float(np.nanpercentile(good, 2))
            vmax = float(np.nanpercentile(good, 98))
            plt.figure(figsize=(6, 6))
            plt.imshow(data, vmin=vmin, vmax=vmax)
            plt.title(title + f" (band {use_band})")
            plt.axis('off')
            out_png = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_quicklook.png")
            plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    except Exception as e:
        warnings.warn(f"Quicklook failed for {os.path.basename(path)}: {e}")


def read_raster_info(path: str) -> RasterInfo:
    name = os.path.basename(path)
    year, month = parse_year_month_from_name(name)
    with rasterio.open(path) as ds:
        crs = ds.crs
        transform = ds.transform
        bounds = ds.bounds
        width, height = ds.width, ds.height
        band_count = ds.count
        band_desc = ds.descriptions or tuple()
        dtypes = [ds.dtypes[i] for i in range(band_count)]
        nodata = [ds.nodatavals[i] for i in range(band_count)]

        px_x = transform.a
        px_y = -transform.e

        if crs and CRS.from_user_input(crs).to_epsg() == 4326:
            cx = (bounds.left + bounds.right) / 2.0
            cy = (bounds.top + bounds.bottom) / 2.0
            px_m_ew, px_m_ns = geodesic_pixel_meters(cx, cy, px_x, px_y)
        else:
            if crs and CRS.from_user_input(crs).axis_info and \
               CRS.from_user_input(crs).axis_info[0].unit_name.lower() in ("metre", "meter"):
                px_m_ew, px_m_ns = abs(px_x), abs(px_y)
            else:
                px_m_ew, px_m_ns = float("nan"), float("nan")

        approx_ok = approx_1km_lataware(crs, (bounds.left, bounds.bottom, bounds.right, bounds.top),
                                        px_x, px_y, px_m_ew, px_m_ns)

        if crs and CRS.from_user_input(crs).to_epsg() != 4326:
            img_poly = box(*bounds)
            img_poly_wgs84 = reproject_polygon(img_poly, crs, CRS.from_epsg(4326))
        else:
            img_poly_wgs84 = box(*bounds)
        overlap_roi, overlap_img = compute_overlap_ratios(img_poly_wgs84, ROI_WGS84)


        align_ok = check_alignment_to_roi_grid(ds, ROI_WGS84)


        if STATS_ON_ROI:
            try:
                stats = stats_on_roi(ds, ROI_WGS84)
            except Exception:
                stats = [read_stats_nan_safe(ds.read(b, masked=False)) for b in range(1, band_count + 1)]
        else:
            stats = [read_stats_nan_safe(ds.read(b, masked=False)) for b in range(1, band_count + 1)]

    return RasterInfo(
        path=path, name=name, year=year, month=month, crs=crs,
        width=width, height=height, transform=transform,
        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
        pixel_size_x=px_x, pixel_size_y=px_y,
        px_m_east_west=px_m_ew, px_m_north_south=px_m_ns,
        approx_1km_ok=approx_ok, band_count=band_count,
        band_descriptions=band_desc, dtypes=dtypes, nodata=nodata,
        stats=stats, overlap_ratio_roi=overlap_roi,
        overlap_ratio_img=overlap_img, align_ok=align_ok
    )

def main():
    tif_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.tif"))) + \
                sorted(glob.glob(os.path.join(DATA_DIR, "*.tiff")))
    if not tif_paths:
        print(".tif/.tiff：", DATA_DIR); return

    rows = []
    for p in tif_paths:
        ri = read_raster_info(p)

        band_stats_str = "; ".join([f"b{i+1}:min={mn:.4g},max={mx:.4g},mean={me:.4g}"
                                    for i, (mn, mx, me) in enumerate(ri.stats)])
        rows.append({
            "file": ri.name,
            "year": ri.year, "month": ri.month,
            "crs": str(ri.crs) if ri.crs else "None",
            "width": ri.width, "height": ri.height,
            "pixel_size_x": ri.pixel_size_x, "pixel_size_y": ri.pixel_size_y,
            "px_m_east_west": ri.px_m_east_west, "px_m_north_south": ri.px_m_north_south,
            "approx_1km_ok": ri.approx_1km_ok,
            "bounds_left": ri.bounds[0], "bounds_bottom": ri.bounds[1],
            "bounds_right": ri.bounds[2], "bounds_top": ri.bounds[3],
            "overlap_ratio_roi": ri.overlap_ratio_roi,
            "overlap_ratio_img": ri.overlap_ratio_img,
            "align_ok_wgs84": ri.align_ok,
            "band_count": ri.band_count,
            "band_descriptions": "; ".join([str(d) for d in ri.band_descriptions]) if ri.band_descriptions else "",
            "dtypes": "; ".join(ri.dtypes),
            "nodata": "; ".join(["None" if v is None else str(v) for v in ri.nodata]),
            "band_stats": band_stats_str
        })

        if MAKE_QUICKLOOK:
            title = f"{ri.name} ({ri.year or ''}{(ri.month or '')})"
            make_quicklook(ri.path, DATA_DIR, title)

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

    print(f"{len(rows)} \nCSV：{CSV_OUT}")
    print("- approx_1km_ok （WGS84  EW≈1000*cos(lat)、NS≈1000）")
    print("- overlap_ratio_roi=1 show ROI align_ok_wgs84=True with ROI ")
    if STATS_ON_ROI:
        print("ROI NaN")


if __name__ == "__main__":
    main()
