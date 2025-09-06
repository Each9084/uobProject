from pathlib import Path
import pandas as pd
import numpy as np

MODEL_DIR = Path(r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\model")
QC_DIR    = MODEL_DIR / "QC"
QC_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = [MODEL_DIR / "merged_2019.csv", MODEL_DIR / "merged_2022.csv"]

# ---------------- Config ----------------
# Range
LOWER_LIMIT = 10.0
UPPER_LIMIT = 37.0
WHITELIST_TOL = 1.5

#reported 42.6
OFFICIAL_HOT_FLOOR = 42.6
OFFICIAL_STATIONS = {"Longchamp", "Luxembourg", "Paris-Montsouris", "Saint-Maur"}
SUMMER_MONTHS = {6, 7, 8}

# Step test

#SAME over 5 delete
STEP_THRESHOLD = 5.0  # °C

# Stuck test 6 hours same
STUCK_WINDOW_HRS = 6
STUCK_TOL = 0.01      # °C

# Soft spatial Robust z-score >3.5 out of the baseline
SPATIAL_SOFT_Z = 3.5

#station num < 3 meaningless
MIN_GROUP_SIZE = 3


#A data frame with uniform cost at the daytime hour level and ordered by station
def load_df(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df["time"] = pd.to_datetime(df["time"], utc=False)  # make sure the time
    hrs = df["time"].dt.hour
    df = df[(hrs >= 6) & (hrs <= 18)].copy()
    df["date"]  = df["time"].dt.date
    df["year"]  = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"]   = df["time"].dt.day
    df["hour"]  = df["time"].dt.hour
    # station → month → day → hour
    df = df.sort_values(["station", "month", "day", "hour"], kind="mergesort").reset_index(drop=True)
    return df

def save(df: pd.DataFrame, out_fp: Path, label: str):
    df.to_csv(out_fp, index=False, encoding="utf-8")
    print(f"[save] {label}: {out_fp}  row={len(df)}  station number={df['station'].nunique()}")

# ---------------- Step 1: Range（含官方背书白名单） ----------------
def qc_step1_range(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    #Make sure it's the same time
    df["time_key"] = df["time"].dt.strftime("%Y-%m-%d %H:%M%z")

    off_mask = df["station"].isin(OFFICIAL_STATIONS)
    off_max = (
        df.loc[off_mask]
          .groupby("time_key")["temperature"]
          .max()
          .rename("off_max_at_time")
    )
    off_min = (
        df.loc[off_mask]
          .groupby("time_key")["temperature"]
          .min()
          .rename("off_min_at_time")
    )

    #caluculate the highest and lowest 4 station temp to make sure the up line and low line
    df = df.merge(off_max, left_on="time_key", right_index=True, how="left")
    df = df.merge(off_min, left_on="time_key", right_index=True, how="left")

    in_basic = (df["temperature"] >= LOWER_LIMIT) & (df["temperature"] <= UPPER_LIMIT)

    # calculate Upper limit whitelist
    over_u = (df["temperature"] > UPPER_LIMIT) & df["month"].isin(SUMMER_MONTHS)
    near_off_high = (np.abs(df["temperature"] - df["off_max_at_time"]) <= WHITELIST_TOL)
    off_hot = (df["off_max_at_time"] >= OFFICIAL_HOT_FLOOR)
    whitelist_high = over_u & (near_off_high | off_hot)

    # Lower limit whitelist
    under_l = (df["temperature"] < LOWER_LIMIT) & df["month"].isin(SUMMER_MONTHS)
    near_off_low = (np.abs(df["temperature"] - df["off_min_at_time"]) <= WHITELIST_TOL)
    whitelist_low = under_l & near_off_low

    keep = in_basic | whitelist_high | whitelist_low
    print(f"[Step1-Range] delete {(~keep).sum()} / {len(df)} ({(~keep).mean():.3%})；whitelist keep high temp {whitelist_high.sum()} 条。")

    return df.loc[keep].drop(columns=["time_key", "off_max_at_time", "off_min_at_time"])

# Step 2: Hourly "jump" inspection(5 degree)
def qc_step2_step(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["station", "time"]).reset_index(drop=True)

    # record front data
    df["prev_time"] = df.groupby("station")["time"].shift(1)
    df["prev_temp"] = df.groupby("station")["temperature"].shift(1)

    # Interval (minutes)
    dt_min = (df["time"] - df["prev_time"]).dt.total_seconds() / 60.0
    same_day_consecutive = dt_min.eq(60.0)

    diff = df["temperature"] - df["prev_temp"]
    bad = same_day_consecutive & (diff.abs() > STEP_THRESHOLD)

    print(f"[Step2-Step] delete {bad.sum()} / {len(df)} ({bad.mean():.3%})。")
    return df.loc[~bad].drop(columns=["prev_time", "prev_temp"])

# Stuck
def qc_step3_stuck(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["station", "time"]).reset_index(drop=True)

    # Calculate whether the adjacent sites are "1 hour continuous"
    dt_min = df.groupby("station")["time"].diff().dt.total_seconds() / 60.0
    contig = dt_min.eq(60.0)
    equal  = df.groupby("station")["temperature"].diff().abs().le(STUCK_TOL)

    # True means "in the same constant segment as the previous one"
    same_seg = contig & equal

    start_new_block = (~same_seg) | same_seg.isna()

    # mark id
    block_id = start_new_block.groupby(df["station"]).cumsum()

    # record size
    block_len = df.groupby(["station", block_id])["temperature"].transform("size")

    stuck_flag = block_len.ge(STUCK_WINDOW_HRS)
    #remove
    removed = int(stuck_flag.sum())
    print(f"[Step3-Stuck] delete {removed} / {len(df)} ({removed/len(df):.3%})。")

    return df.loc[~stuck_flag]

# Step 4 Record-by-record spatial outliers
def qc_step4_spatial_soft(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # For each (date, hour) group, find the median and MAD (median absolute deviation)
    stats = (
        df.groupby(["date", "hour"])["temperature"]
          .agg(
              median="median",
              mad=lambda x: float(np.median(np.abs(x - np.median(x)))),
              std=lambda x: float(x.std(ddof=1)) if x.size > 1 else np.nan,
              n="count"
          )
          .reset_index()
    )
    df = df.merge(stats, on=["date", "hour"], how="left")

    # from mad to calculate the z score
    sigma = 1.4826 * df["mad"]
    sigma = np.where((df["mad"] <= 0) | df["mad"].isna(), df["std"], sigma)
    valid_grp = df["n"] >= MIN_GROUP_SIZE

    # z score
    z = np.abs((df["temperature"] - df["median"]) / sigma)
    z = np.where((pd.Series(sigma) > 0) & valid_grp, z, 0.0)

    flagged = pd.Series(z > SPATIAL_SOFT_Z)

    # compare withe the witelist
    df["time_key"] = df["time"].dt.strftime("%Y-%m-%d %H:%M%z")
    off_mask = df["station"].isin(OFFICIAL_STATIONS)
    off_max = df.loc[off_mask].groupby("time_key")["temperature"].max().rename("off_max_at_time")
    off_min = df.loc[off_mask].groupby("time_key")["temperature"].min().rename("off_min_at_time")
    df = df.merge(off_max, left_on="time_key", right_index=True, how="left")
    df = df.merge(off_min, left_on="time_key", right_index=True, how="left")

    keep_by_official_high = (np.abs(df["temperature"] - df["off_max_at_time"]) <= WHITELIST_TOL) | (df["off_max_at_time"] >= OFFICIAL_HOT_FLOOR)
    keep_by_official_low  = (np.abs(df["temperature"] - df["off_min_at_time"]) <= WHITELIST_TOL)

    keep = (~flagged) | keep_by_official_high | keep_by_official_low

    removed = int((~keep).sum())
    print(f"[Step4-SpatialSoft] delete {removed} / {len(df)} ({removed/len(df):.3%})；")

    out = (
        df.loc[keep, df.columns.difference(["median","mad","std","n","time_key","off_max_at_time","off_min_at_time"])]
          .sort_values(["station", "month", "day", "hour"], kind="mergesort")
          .reset_index(drop=True)
    )
    return out

# ---------------- Main ----------------
def run_one(fp: Path):
    year = fp.stem[-4:]
    print(f"\n {year}")
    df0 = load_df(fp)

    s1 = qc_step1_range(df0)
    save(s1, QC_DIR / f"merged_{year}_qc_step1_range.csv", f"{year} Step1")

    s2 = qc_step2_step(s1)
    save(s2, QC_DIR / f"merged_{year}_qc_step2_step.csv", f"{year} Step2")

    s3 = qc_step3_stuck(s2)
    save(s3, QC_DIR / f"merged_{year}_qc_step3_stuck.csv", f"{year} Step3")

    s4 = qc_step4_spatial_soft(s3)
    save(s4, QC_DIR / f"merged_{year}_qc_step4_spatial_soft.csv", f"{year} Step4-soft")

    # result
    save(s4, QC_DIR / f"merged_{year}_qc_final.csv", f"{year} FINAL")

def main():
    pd.set_option("display.width", 160)
    for fp in INPUTS:
        if fp.exists():
            run_one(fp)
        else:
            print(f"skip: {fp}")

if __name__ == "__main__":
    main()
