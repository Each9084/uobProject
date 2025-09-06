
import os
from pprint import pprint

import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform_bounds


DATA_DIR = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\GEE\slow"

FILE_WHITELIST = None

# 目标 BBOX（WGS84）
TARGET_BBOX_WGS84 = dict(
    minLon=2.194225, minLat=48.785758,
    maxLon=2.497260, maxLat=48.931653
)


N_COLS = 24
N_ROWS = 17
dx = (TARGET_BBOX_WGS84["maxLon"] - TARGET_BBOX_WGS84["minLon"]) / N_COLS
dy = (TARGET_BBOX_WGS84["maxLat"] - TARGET_BBOX_WGS84["minLat"]) / N_ROWS

XFORM_EXPECTED = Affine(dx, 0, TARGET_BBOX_WGS84["minLon"],
                        0, -dy, TARGET_BBOX_WGS84["maxLat"])


TOL_DEG_BBOX = 1e-10
TOL_AFFINE   = 1e-12
# ===========================================


def aff_almost_equal(a: Affine, b: Affine, tol=TOL_AFFINE) -> bool:
    return (abs(a.a - b.a) <= tol and
            abs(a.b - b.b) <= tol and
            abs(a.c - b.c) <= tol and
            abs(a.d - b.d) <= tol and
            abs(a.e - b.e) <= tol and
            abs(a.f - b.f) <= tol)

def almost_equal(a, b, tol=TOL_DEG_BBOX) -> bool:
    return abs(a - b) <= tol

def check_file(path, xform_expected: Affine, bbox_target: dict):
    with rasterio.open(path) as ds:
        info = {
            "path": path,
            "name": os.path.basename(path),
            "crs": str(ds.crs) if ds.crs else None,
            "width": ds.width,
            "height": ds.height,
            "transform": ds.transform,
        }

        #CRS  EPSG:4326
        crs_ok = (ds.crs is not None and ds.crs.to_string().upper() in ("EPSG:4326", "WGS84"))
        info["crs_ok"] = crs_ok

        # 24x17
        size_ok = (ds.width == N_COLS and ds.height == N_ROWS)
        info["size_ok"] = size_ok

        #
        xform_ok = aff_almost_equal(ds.transform, xform_expected)
        info["xform_ok"] = xform_ok

        # BBOX 一
        # EPSG:4326
        bounds = ds.bounds
        bounds_ok = (almost_equal(bounds.left,   bbox_target["minLon"]) and
                     almost_equal(bounds.right,  bbox_target["maxLon"]) and
                     almost_equal(bounds.bottom, bbox_target["minLat"]) and
                     almost_equal(bounds.top,    bbox_target["maxLat"]))
        info["bounds"] = dict(minLon=bounds.left, minLat=bounds.bottom,
                              maxLon=bounds.right, maxLat=bounds.top)
        info["bounds_ok"] = bounds_ok

        # summary
        info["all_ok"] = all([crs_ok, size_ok, xform_ok, bounds_ok])
        return info

def main():
    # collect file
    if FILE_WHITELIST:
        files = [os.path.join(DATA_DIR, f) for f in FILE_WHITELIST]
    else:
        files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
                 if f.lower().endswith(".tif") or f.lower().endswith(".tiff")]
    files.sort()

    print("="*80)
    print("WGS84 Fixed Grid Consistency Check")
    print(f"Folder: {DATA_DIR}")
    print(f"Target BBOX: {TARGET_BBOX_WGS84}")
    print(f"Expected size: {N_COLS} x {N_ROWS}")
    print(f"Expected transform: {XFORM_EXPECTED}")
    print("="*80)

    if not files:
        print("unfound GeoTIFF。")
        return

    # check the step
    results = []
    for fp in files:
        try:
            res = check_file(fp, XFORM_EXPECTED, TARGET_BBOX_WGS84)
        except Exception as e:
            res = {"path": fp, "name": os.path.basename(fp), "error": str(e)}
        results.append(res)

    # print
    global_ok = True
    ref_transform = None
    same_transform_ok = True

    for r in results:
        print("\n---", r.get("name", "(unknown)"), "---")
        if "error" in r:
            print("ERROR:", r["error"])
            global_ok = False
            continue

        print("CRS:", r["crs"])
        print("Size (W x H):", r["width"], "x", r["height"])
        print("Transform:", r["transform"])
        print("Bounds (WGS84):"); pprint(r["bounds"])

        print("Checks:",
              f"crs_ok={r['crs_ok']}, size_ok={r['size_ok']},",
              f"xform_ok={r['xform_ok']}, bounds_ok={r['bounds_ok']}")
        print("ALL OK? ->", r["all_ok"])

        if not r["all_ok"]:
            global_ok = False

        # record
        if r.get("all_ok"):
            if ref_transform is None:
                ref_transform = r["transform"]
            else:
                if not aff_almost_equal(ref_transform, r["transform"]):
                    same_transform_ok = False

    print("\n" + "="*80)
    if global_ok and same_transform_ok:
        print("all ok")
    elif global_ok and not same_transform_ok:
        print("success")
    else:
        print("erro")
    print("="*80)


if __name__ == "__main__":
    main()
