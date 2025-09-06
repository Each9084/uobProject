
import os, re, math
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds


BASE_DIR = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\GEE\fast"

ROI_LON_MIN, ROI_LON_MAX = 2.15, 2.55
ROI_LAT_MIN, ROI_LAT_MAX = 48.75, 48.95
PIX_TOL_DEG = 0.009
RES_TOL_DEG = 0.001

MISSING_RATIO_ALERT = 0.10  # 缺失比例>10%告警

T2M_MIN, T2M_MAX = -50.0, 60.0      # ℃
SW_MIN,  SW_MAX  = -0.05, 1.20      # m3/m3

BAND_RE = re.compile(
    r"^(?P<utc_date>\d{8})T(?P<utc_hour>\d{2})_(?P<var>t2m|swvl1)_(?P<loc_date>\d{8})_(?P<loc_hour>\d{2})$"
)

def local_to_utc_hour(year: int, month: int, day: int, local_hour: int) -> Tuple[int, str]:

    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Paris")
        dt_local = datetime(year, month, day, local_hour, 0, 0, tzinfo=tz)
        dt_utc   = dt_local.astimezone(ZoneInfo("UTC"))
        return dt_utc.hour, "zoneinfo"
    except Exception:
        try:
            import pytz
            tz = pytz.timezone("Europe/Paris")
            dt_local = tz.localize(datetime(year, month, day, local_hour, 0, 0))
            dt_utc   = dt_local.astimezone(pytz.utc)
            return dt_utc.hour, "pytz"
        except Exception:

            return (local_hour - 2) % 24, "fallback(+2)"

def parse_band_name(nm: str) -> Dict:
    m = BAND_RE.match(nm or "")
    if not m:
        return {}
    d = m.groupdict()
    d["utc_hour"] = int(d["utc_hour"])
    d["loc_hour"] = int(d["loc_hour"])
    return d

def qc_one_file(path: str) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    fn = os.path.basename(path)
    per_band_rows: List[Dict] = []
    red_rows: List[Dict] = []

    per_file = {
        "file": fn,
        "crs_ok": False,
        "res_ok": False,
        "bounds_ok": False,
        "dtype_ok": False,
        "band_total": None,
        "t2m_bands": 0,
        "swvl1_bands": 0,
        "t2m_missing_ratio_mean": np.nan,
        "swvl1_missing_ratio_mean": np.nan,
        "notes": []
    }

    try:
        with rasterio.open(path) as src:
            # basic info
            crs = src.crs.to_string() if src.crs else None
            resx, resy = src.res
            b = src.bounds
            lon_min, lat_min, lon_max, lat_max = b.left, b.bottom, b.right, b.top
            per_file["band_total"] = src.count
            per_file["dtype_ok"] = set(src.dtypes) == {"float32"}


            per_file["crs_ok"] = (crs == "EPSG:4326")
            if not per_file["crs_ok"]:
                per_file["notes"].append(f"CRS={crs}")

            per_file["res_ok"] = (abs(resx - 0.00898315284) <= RES_TOL_DEG and
                                  abs(resy - 0.00898315284) <= RES_TOL_DEG)
            if not per_file["res_ok"]:
                per_file["notes"].append(f"res=({resx:.6f},{resy:.6f})")

            per_file["bounds_ok"] = \
                (ROI_LON_MIN - PIX_TOL_DEG) <= lon_min <= (ROI_LON_MIN + PIX_TOL_DEG) and \
                (ROI_LON_MAX - PIX_TOL_DEG) <= lon_max <= (ROI_LON_MAX + PIX_TOL_DEG) and \
                (ROI_LAT_MIN - PIX_TOL_DEG) <= lat_min <= (ROI_LAT_MIN + PIX_TOL_DEG) and \
                (ROI_LAT_MAX - PIX_TOL_DEG) <= lat_max <= (ROI_LAT_MAX + PIX_TOL_DEG)
            if not per_file["bounds_ok"]:
                per_file["notes"].append(f"bounds=({lon_min:.6f},{lat_min:.6f},{lon_max:.6f},{lat_max:.6f})")

            names = list(src.descriptions)
            width, height = src.width, src.height
            total_px = width * height

            t2m_miss = []
            sw_miss  = []

            for i, nm in enumerate(names, start=1):
                info = parse_band_name(nm)
                if not info:
                    per_file["notes"].append(f"bad_name:{nm}")
                    continue
                var = info["var"]
                # UTC↔local time
                y_loc = int(info["loc_date"][0:4]); m_loc = int(info["loc_date"][4:6]); d_loc = int(info["loc_date"][6:8])
                utc_hour_should, tz_src = local_to_utc_hour(y_loc, m_loc, d_loc, info["loc_hour"])
                utc_ok = (utc_hour_should == info["utc_hour"])
                if not utc_ok:
                    per_file["notes"].append(f"UTC_mismatch:{nm} expect {utc_hour_should:02d} via {tz_src}")

                arr = src.read(i, masked=True).astype("float64")  # masked array
                # static
                is_masked = np.ma.getmaskarray(arr)
                valid_px  = int(np.sum(~is_masked))
                missing_px= int(np.sum(is_masked))
                missing_ratio = missing_px / float(total_px) if total_px else np.nan

                data = arr.compressed()
                nan_count = int(np.isnan(data).sum())
                # extreme check
                inf_count = int(np.isinf(data).sum())

                vmin = float(np.min(data)) if data.size else np.nan
                vmax = float(np.max(data)) if data.size else np.nan
                vmean= float(np.mean(data)) if data.size else np.nan
                vstd = float(np.std(data))  if data.size else np.nan

                # redline
                if missing_ratio > MISSING_RATIO_ALERT:
                    red_rows.append({
                        "file": fn, "band": nm, "issue": "missing_ratio_gt_threshold",
                        "missing_ratio": missing_ratio
                    })

                if var == "t2m":
                    if not (math.isnan(vmin) or (T2M_MIN <= vmin <= T2M_MAX)):
                        red_rows.append({"file": fn, "band": nm, "issue": "t2m_min_out_of_range", "value": vmin})
                    if not (math.isnan(vmax) or (T2M_MIN <= vmax <= T2M_MAX)):
                        red_rows.append({"file": fn, "band": nm, "issue": "t2m_max_out_of_range", "value": vmax})
                    t2m_miss.append(missing_ratio)
                else:
                    if not (math.isnan(vmin) or (SW_MIN <= vmin <= SW_MAX)):
                        red_rows.append({"file": fn, "band": nm, "issue": "swvl1_min_out_of_range", "value": vmin})
                    if not (math.isnan(vmax) or (SW_MIN <= vmax <= SW_MAX)):
                        red_rows.append({"file": fn, "band": nm, "issue": "swvl1_max_out_of_range", "value": vmax})
                    sw_miss.append(missing_ratio)

                per_band_rows.append({
                    "file": fn,
                    "band": nm,
                    "var": var,
                    "utc_date": info["utc_date"],
                    "utc_hour": info["utc_hour"],
                    "local_date": info["loc_date"],
                    "local_hour": info["loc_hour"],
                    "utc_ok": utc_ok,
                    "width": width, "height": height, "total_px": total_px,
                    "valid_px": valid_px, "missing_px": missing_px, "missing_ratio": missing_ratio,
                    "nan_count": nan_count, "inf_count": inf_count,
                    "min": vmin, "max": vmax, "mean": vmean, "std": vstd
                })

            per_file["t2m_bands"] = sum(1 for n in names if "_t2m_" in (n or ""))
            per_file["swvl1_bands"] = sum(1 for n in names if "_swvl1_" in (n or ""))
            per_file["t2m_missing_ratio_mean"] = float(np.nanmean(t2m_miss)) if len(t2m_miss) else np.nan
            per_file["swvl1_missing_ratio_mean"] = float(np.nanmean(sw_miss)) if len(sw_miss) else np.nan

    except Exception as e:
        per_file["notes"].append(f"open_error:{e}")

    per_band_df = pd.DataFrame(per_band_rows)
    redflags_df = pd.DataFrame(red_rows)
    return per_band_df, per_file, redflags_df

def main():
    all_band_rows = []
    all_file_rows = []
    all_red_rows  = []

    files = sorted([
        fn for fn in os.listdir(BASE_DIR)
        if re.match(r"^fastvars_\d{8}_local0618_modis1km\.tif$", fn)
    ])
    if not files:
        print("No matching files found.")
        return

    for fn in files:
        per_band_df, per_file_dict, red_df = qc_one_file(os.path.join(BASE_DIR, fn))
        all_file_rows.append(per_file_dict)
        if not per_band_df.empty:
            all_band_rows.append(per_band_df)
        if not red_df.empty:
            all_red_rows.append(red_df)

    band_df = pd.concat(all_band_rows, ignore_index=True) if all_band_rows else pd.DataFrame()
    file_df = pd.DataFrame(all_file_rows)
    red_df  = pd.concat(all_red_rows, ignore_index=True) if all_red_rows else pd.DataFrame()


    band_csv = os.path.join(BASE_DIR, "fastvars_qc_bandstats.csv")
    file_csv = os.path.join(BASE_DIR, "fastvars_qc_detailed_summary.csv")
    red_csv  = os.path.join(BASE_DIR, "fastvars_qc_redflags.csv")

    band_df.to_csv(band_csv, index=False, encoding="utf-8-sig")
    file_df.to_csv(file_csv, index=False, encoding="utf-8-sig")
    red_df.to_csv(red_csv, index=False, encoding="utf-8-sig")

    print("Saved:", band_csv)
    print("Saved:", file_csv)
    print("Saved:", red_csv)
    if not band_df.empty:
        print("Median missing ratio (t2m):",
              band_df.loc[band_df["var"]=="t2m","missing_ratio"].median())
        print("Median missing ratio (swvl1):",
              band_df.loc[band_df["var"]=="swvl1","missing_ratio"].median())

if __name__ == "__main__":
    main()
