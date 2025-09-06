
import pandas as pd
from pathlib import Path
import xarray as xr
import re


IN_ROOT = Path(r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\aeris\unzipped")
OUT_DIR = Path(r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\aeris\aerisOutput")
OUT_DIR.mkdir(parents=True, exist_ok=True)


WANTED_DIRS = {
    "LONGCHAMP_MTO-1H_2025-08-10",
    "LUXEMBOURG_MTO-1H_2025-08-10",
    "PARIS-MONTSOURIS_MTO-1H_2025-08-10",
    "ST-MAUR_MTO-1H_2025-08-12",
}

YEARS  = {2019, 2022}
MONTHS = {6, 7, 8}
HOUR_MIN, HOUR_MAX = 6, 18

# station id
STATION_NAME_MAP = {
    "75116008": "Longchamp",
    "75106001": "Luxembourg",
    "75114001": "Paris-Montsouris",
    "94068001": "Saint-Maur",
}

# NetCDF
def open_dataset_any_engine(path: Path) -> xr.Dataset:
    """netcdf4 → h5netcdf → scipy"""
    for eng in ("netcdf4", "h5netcdf", "scipy"):
        try:
            return xr.open_dataset(path, engine=eng)
        except Exception:
            continue
    raise RuntimeError(f" netcdf4/h5netcdf/scipy : {path}")

def pick_temperature_var(ds: xr.Dataset) -> xr.DataArray:
    """ CF standard_name=air_temperature """
    if "ta" in ds.variables:
        return ds["ta"]
    for v in ds.variables:
        if ds[v].attrs.get("standard_name", "").lower() == "air_temperature":
            return ds[v]
    raise KeyError(" standard_name=air_temperature")

def infer_station_id_from_filename(nc_path: Path) -> str:
    m = re.match(r"(\d+)_", nc_path.name)
    return m.group(1) if m else ""

def infer_station_name(nc_path: Path, ds: xr.Dataset, station_id: str) -> str:

    if station_id and station_id in STATION_NAME_MAP:
        return STATION_NAME_MAP[station_id]


    cand = ds.attrs.get("station_name") or ds.attrs.get("site_name") or ds.attrs.get("platform_name")
    if isinstance(cand, str) and cand.strip():
        return cand.strip()


    if "station_name" in ds.variables:
        try:
            val = ds["station_name"].values
            if hasattr(val, "item"):
                val = val.item()
            if isinstance(val, bytes):
                val = val.decode("utf-8", "ignore")
            name = str(val).strip()

            name = re.sub(r"\(.*?id\s*=\s*\d+.*?\)", "", name, flags=re.I).strip()
            if name:
                return name
        except Exception:
            pass



    tokens = nc_path.stem.split("_")
    if len(tokens) >= 3:
        try:
            name_tokens = []
            for tk in tokens[1:]:
                if tk.upper().startswith("MTO"):
                    break
                name_tokens.append(tk)
            raw = " ".join(name_tokens)

            pretty = " ".join([p.title() for p in raw.split()])
            # "St-Maur"  "Saint-Maur"
            pretty = pretty.replace("St-Maur", "Saint-Maur")
            return pretty or station_id or nc_path.stem
        except Exception:
            pass

    return station_id or nc_path.stem


def process_nc(nc_path: Path) -> pd.DataFrame:
    ds = open_dataset_any_engine(nc_path)
    lat = float(ds["lat"].values)
    lon = float(ds["lon"].values)
    station_id = infer_station_id_from_filename(nc_path)
    station_name = infer_station_name(nc_path, ds, station_id)

    # AERIS  UTC
    time_utc = pd.to_datetime(ds["time"].values, utc=True)
    time_local = time_utc.tz_convert("Europe/Paris")  # 自动处理夏令时

    # （K → °C）
    taK = pick_temperature_var(ds).values
    tempC = pd.Series(taK, dtype="float64") - 273.15

    df = pd.DataFrame({
        "station": station_name,
        "station_id": station_id,
        "time": time_local,
        "lat": lat,
        "lon": lon,
        "temperature": tempC.values
    })

    df = df[df["time"].dt.month.isin(MONTHS) & df["time"].dt.hour.between(HOUR_MIN, HOUR_MAX)]

    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df = df.dropna(subset=["temperature"])

    # order
    return df.sort_values("time").reset_index(drop=True)


def main():
    all_rows = []

    for sub in sorted(IN_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name not in WANTED_DIRS:
            continue

        for yr in YEARS:
            for nc in sorted(sub.glob(f"*_{yr}.nc")):
                print(f"read: {nc}")
                part = process_nc(nc)
                all_rows.append(part)

    if not all_rows:
        raise RuntimeError("no NetCDF ")

    all_df = pd.concat(all_rows, ignore_index=True)

    all_df = all_df[
        all_df["time"].dt.year.isin(YEARS) &
        all_df["time"].dt.month.isin(MONTHS) &
        all_df["time"].dt.hour.between(HOUR_MIN, HOUR_MAX)
    ].copy()


    for yr in sorted(YEARS):
        sub = all_df[all_df["time"].dt.year == yr].sort_values(["station", "time"]).reset_index(drop=True)
        if sub.empty:
            continue

        # summer +0200（CEST）
        offs = sub["time"].dt.strftime("%z").dropna().unique().tolist()
        if set(offs) != {"+0200"}:
            print(f"[warn] {yr}  +0200：{offs} 6–8 ，+0200）")

        out_path = OUT_DIR / f"{yr}_AERIS.csv"
        sub.to_csv(out_path, index=False)
        print(f"save: {out_path}  row num: {len(sub)}")

if __name__ == "__main__":
    pd.set_option("display.width", 140)
    main()
