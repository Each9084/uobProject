
import os
import re
import math
import numpy as np
import rasterio


OUT_DIR = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\comephore\updateDir"
REF_PATH = r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\GEE\slow\201906.tif"


DAILY_PATTERN = re.compile(r"rainvars_(\d{8})_local06to18_slow\.tif$", re.IGNORECASE)
BANDNAME_PATTERN = re.compile(r"^rain_(\d{8})T(0\d|1\d)$")

LOCAL_HOURS_EXPECT = list(range(6, 19))

# ===== 读取参考网格信息 =====
with rasterio.open(REF_PATH) as ref:
    REF_CRS = ref.crs
    REF_TRANSFORM = ref.transform
    REF_WIDTH = ref.width
    REF_HEIGHT = ref.height
    REF_RES = ref.res

def almost_equal(a, b, eps=1e-7):
    return abs(a - b) <= eps

def transform_close(t1, t2, eps=1e-6):

    vals1 = (t1.a, t1.b, t1.c, t1.d, t1.e, t1.f)
    vals2 = (t2.a, t2.b, t2.c, t2.d, t2.e, t2.f)
    return all(almost_equal(x, y, eps) for x, y in zip(vals1, vals2))

def print_header(title):
    bar = "═" * max(10, len(title) + 2)
    print(f"\n{bar}\n {title}\n{bar}")


print_header("grid info")
print(f"CRS:         {REF_CRS}")
print(f"size:        {REF_WIDTH} x {REF_HEIGHT}")
print(f"projection:      {REF_RES}")
print(f"Transform:   {REF_TRANSFORM}")

files = [f for f in os.listdir(OUT_DIR) if f.lower().endswith(".tif")]
files = sorted(files)

if not files:
    print_header("result")
    print("no .tif file")
    raise SystemExit

total = 0
ok_count = 0
warn_count = 0
fail_count = 0

for fn in files:
    total += 1
    m = DAILY_PATTERN.match(fn)
    if not m:
        print_header(f"{fn}")
        print("  rainvars_YYYYMMDD_local06to18_slow.tif")
        warn_count += 1
        continue

    day_str = m.group(1)  # local time（YYYYMMDD）
    path = os.path.join(OUT_DIR, fn)

    with rasterio.open(path) as ds:
        errs = []
        warns = []


        if ds.crs != REF_CRS:
            errs.append(f"CRS  {REF_CRS}， {ds.crs}")

        if not transform_close(ds.transform, REF_TRANSFORM):
            errs.append("Transform ")

        if (ds.width != REF_WIDTH) or (ds.height != REF_HEIGHT):
            errs.append(f" {REF_WIDTH}x{REF_HEIGHT}， {ds.width}x{ds.height}")

        # 2) 波段数、dtype、nodata
        if ds.count != 13:
            errs.append(f" 13， {ds.count}")

        if any(dt != "float32" for dt in ds.dtypes):
            errs.append(f"dtype  float32， {ds.dtypes}")

        if ds.nodata is None or not almost_equal(float(ds.nodata), -9999.0, eps=1e-9):
            errs.append(f"nodata  -9999.0， {ds.nodata}")

        band_hours = []
        day_mismatch = False
        missing_hours = []
        nodata_stats = []  # (hour, nodata_pct)

        for i in range(1, ds.count + 1):
            desc = ds.descriptions[i - 1] or ""
            mm = BANDNAME_PATTERN.match(desc)
            if not mm:
                errs.append(f"{i}rain_YYYYMMDDTHH：'{desc}'")
                continue

            b_day = mm.group(1)
            b_hh = int(mm.group(2))
            if b_day != day_str:
                day_mismatch = True
            band_hours.append(b_hh)

            # calculate nodata
            arr = ds.read(i, masked=False)
            total_px = arr.size
            nodata_px = int(np.sum(arr == ds.nodata))
            nodata_pct = (nodata_px / total_px) * 100.0
            nodata_stats.append((b_hh, nodata_pct))

        # check hour
        expected_set = set(LOCAL_HOURS_EXPECT)
        got_set = set(band_hours)
        if got_set != expected_set:
            missing = sorted(list(expected_set - got_set))
            extra = sorted(list(got_set - expected_set))
            if missing:
                errs.append(f"missing：{missing}")
            if extra:
                errs.append(f"06..18）：{extra}")

        if day_mismatch:
            errs.append("error")

    # result
    print_header(fn)
    if errs:
        print("error：")
        for e in errs:
            print("   - " + e)
        fail_count += 1
    else:
        print("ok")
        ok_count += 1

    # print
    print("basic info ")
    with rasterio.open(path) as ds:
        print(f"  CRS:       {ds.crs}")
        print(f"  size:      {ds.width} x {ds.height}")
        print(f"  projection:    {ds.res}")
        print(f"  num:    {ds.count}")
        print(f"  dtype:     {ds.dtypes[0]}")
        print(f"  nodata:    {ds.nodata}")

        is_tags = ds.tags(ns='IMAGE_STRUCTURE')
        comp = is_tags.get('COMPRESSION') or ds.tags().get('COMPRESSION') or ds.tags().get('COMPRESS')
        if comp:
            print(f"  zip:      {comp}")

        # nodata percentage
        names = [(i, ds.descriptions[i-1] or "") for i in range(1, ds.count+1)]
    # nodata
    try:
        nodata_stats_sorted = sorted(nodata_stats, key=lambda x: x[0])
        nodata_line = "  Nodata%:  " + ", ".join([f"{hh:02d}: {pct:.1f}%" for hh, pct in nodata_stats_sorted])
        print("" + ", ".join([n for _, n in names]))
        print(nodata_line)
    except Exception:
        pass


