
import pandas as pd
from pathlib import Path


DATA_DIR = Path(r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\InfoclimatResource")
OUT_DIR  = DATA_DIR / "InfoclimatResourceOutput"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "201906.csv", "201907.csv", "201908.csv",
    "202206.csv", "202207.csv", "202208.csv",
]

# 4 stations
KEEP_STATIONS = {"000BV", "000CT", "000EW", "ME098"}

# lom lat
STATION_COORDS = {
    "000BV": (48.847, 2.356),  # Paris 5ème - Tour Zamansky
    "000CT": (48.852, 2.335),  # Paris 6ème - Saint Germain des Prés
    "000EW": (48.848, 2.409),  # Paris 20ème - Porte de Vincennes
    "ME098": (48.846, 2.267),  # [MAE] ESPE - PARIS
}


YEARS  = {2019, 2022}
MONTHS = {6, 7, 8}
HOUR_MIN, HOUR_MAX = 6, 18



def load_month(fp: Path) -> pd.DataFrame:
    """

    - station_id == 'string'
    - station_id, dh_utc, temperature
    - dh_utc UTC  →  Europe/Paris
    - temperature
    """
    df = pd.read_csv(fp, sep=";", comment="#", dtype=str, encoding="utf-8", engine="python")

    if "station_id" not in df.columns or "dh_utc" not in df.columns or "temperature" not in df.columns:
        raise ValueError(f"warn: {fp}")

    df = df[df["station_id"] != "string"].copy()
    df = df[["station_id", "dh_utc", "temperature"]].copy()

    df = df[df["station_id"].isin(KEEP_STATIONS)].copy()
    if df.empty:
        return df

    # UTC -> Europe/Paris
    df["time_utc"] = pd.to_datetime(df["dh_utc"], format="%Y-%m-%d %H:%M:%S", utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"])
    df["time_local"] = df["time_utc"].dt.tz_convert("Europe/Paris")

    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")

    return df


def main():
    # 合并 6 个文件
    frames = []
    for name in FILES:
        fp = DATA_DIR / name
        if not fp.exists():
            raise FileNotFoundError(f"error: {fp}")
        frames.append(load_month(fp))

    raw = pd.concat(frames, ignore_index=True)
    if raw.empty:
        raise RuntimeError("null")

    raw = raw.set_index("time_local")
    hourly = (
        raw.groupby("station_id")
           .resample("1h")
           .agg({"temperature": "mean"})
           .reset_index()
    )

    hourly["year"]  = hourly["time_local"].dt.year
    hourly["month"] = hourly["time_local"].dt.month
    hourly["hour"]  = hourly["time_local"].dt.hour

    hourly = hourly[
        hourly["year"].isin(YEARS) &
        hourly["month"].isin(MONTHS) &
        (hourly["hour"] >= HOUR_MIN) &
        (hourly["hour"] <= HOUR_MAX)
    ].copy()

    hourly["lat"] = hourly["station_id"].map(lambda s: STATION_COORDS.get(s, (None, None))[0])
    hourly["lon"] = hourly["station_id"].map(lambda s: STATION_COORDS.get(s, (None, None))[1])

    final = hourly.rename(columns={
        "station_id": "station",
        "time_local": "time"
    })[["station", "time", "lat", "lon", "temperature"]].copy()

    # order
    final = final.sort_values(["station", "time"]).reset_index(drop=True)

    # 2019 & 2022
    for yr in [2019, 2022]:
        sub = final[final["time"].dt.year == yr].copy()

        if sub.empty:
            print(f"[警告] {yr} 无记录，跳过导出。")
            continue

        offs = sub["time"].dt.strftime("%z").dropna().unique().tolist()
        assert set(offs) == {"+0200"}, f"{yr} 发现非 +0200 偏移: {offs}"

        out_path = OUT_DIR / f"{yr}_Infoclimat.csv"
        sub.to_csv(out_path, index=False)

        print(f"save: {out_path}")
        print("num:", len(sub))
        print("station num:")
        print(sub.groupby("station")["time"].count())
        print("-" * 60)


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    main()
