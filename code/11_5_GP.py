from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

#import the gp
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from gplearn.fitness import make_fitness


# GP: gplearn
from gplearn.genetic import SymbolicRegressor

# ---------------- Paths ----------------
BASE = Path(r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\model")
QC_DIR = BASE / "QC"
STEPWISE_DIR = BASE / "Stepwise"
OUTDIR = BASE / "GP_CV"
OUTDIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    2019: {
        "data": QC_DIR / "merged_2019_qc_final.csv",
        "groups": STEPWISE_DIR / "groups_from_beta_2019.json",
    },
    2022: {
        "data": QC_DIR / "merged_2022_qc_final.csv",
        "groups": STEPWISE_DIR / "groups_from_beta_2022.json",
    },
}

TARGET = "temperature"

# all column
ALL_PRED_COLS = ["t2m","emis31","ndvi","mndwi","elevation","soil_moisture","rain"]

# set GP config
RANDOM_SEED = 66
KFOLD = 5

# use the mse Goal: The low the better
def _mse_func(y, y_pred, sample_weight):
    if sample_weight is None:
        return float(np.mean((y - y_pred) ** 2))
    return float(np.average((y - y_pred) ** 2, weights=sample_weight))

MSE_FITNESS = make_fitness(function=_mse_func, greater_is_better=False, wrap=False)


GP_PARAMS = dict(
    population_size=1000,
    generations=25,
    tournament_size=20,
    stopping_criteria=0.0,
    const_range=(-1.0, 1.0),
    function_set=("add","sub","mul","div","sqrt","log","abs","neg"),
    metric=MSE_FITNESS,   # mse what we used
    parsimony_coefficient=0.001,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=0,
)



def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def prepare_xy(df: pd.DataFrame, features: list):
    df2 = df[[TARGET] + features].dropna(how="any").reset_index(drop=True)
    X = df2[features].astype(float).values
    y = df2[TARGET].astype(float).values
    return X, y, df2

def run_year(year: int, data_fp: Path, groups_fp: Path):
    print(f"\n=== Year {year} ===")
    if not data_fp.exists() or not groups_fp.exists():
        print(f"[Skip] Missing: {data_fp if not data_fp.exists() else groups_fp}")
        return

    # read aqc nd group
    df = pd.read_csv(data_fp)
    with open(groups_fp, "r", encoding="utf-8") as f:
        group_cfg = json.load(f)
    groups = group_cfg["groups"]  # { "1": [...], "2": [...], ... } or keys as ints
    groups = {int(k): v for k, v in groups.items()}

    # save
    folds_detail = []
    group_summary = []
    #expression
    best_models_txt = []

    best_models_json = []

    #Group(G1..G6)
    for gid in sorted(groups.keys()):
        features = groups[gid]

        features = [c for c in features if c in df.columns]
        X_all, y_all, df_used = prepare_xy(df, features)
        n = len(y_all)
        print(f"[Group {gid}] features={features}  n={n}")

        # KFold
        kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)

        # Tracking the representative model with the best validation RMSE
        best_val_rmse = np.inf
        best_prog_str = None
        best_prog_len = None
        best_prog_depth = None

        fold_idx = 0
        for train_idx, val_idx in kf.split(X_all, y_all):
            fold_idx += 1
            X_tr, X_va = X_all[train_idx], X_all[val_idx]
            y_tr, y_va = y_all[train_idx], y_all[val_idx]

            est = SymbolicRegressor(**GP_PARAMS)
            est.fit(X_tr, y_tr)

            # train
            y_tr_hat = est.predict(X_tr)
            y_va_hat = est.predict(X_va)

            # Verification indicators
            r2_tr = r2_score(y_tr, y_tr_hat)
            r2_va = r2_score(y_va, y_va_hat)
            mae_tr = mean_absolute_error(y_tr, y_tr_hat)
            mae_va = mean_absolute_error(y_va, y_va_hat)
            rmse_tr = rmse(y_tr, y_tr_hat)
            rmse_va = rmse(y_va, y_va_hat)

            # Savebook Details
            folds_detail.append({
                "year": year,
                "group": gid,
                "features": ",".join(features),
                "fold": fold_idx,
                "R2_train": r2_tr,
                "R2_val": r2_va,
                "RMSE_train": rmse_tr,
                "RMSE_val": rmse_va,
                "MAE_train": mae_tr,
                "MAE_val": mae_va,
                "n_train": len(y_tr),
                "n_val": len(y_va),
                "program": str(est._program),
                "program_length": int(est._program.length_),
                "program_depth": int(est._program.depth_),
            })

            # Representative model: Take the fold with the smallest validation RMSE
            if rmse_va < best_val_rmse:
                best_val_rmse = rmse_va
                best_prog_str = str(est._program)
                best_prog_len = int(est._program.length_)
                best_prog_depth = int(est._program.depth_)

        # Summarize the mean ± std of the group
        df_fd = pd.DataFrame([d for d in folds_detail if d["year"]==year and d["group"]==gid])
        summary_row = {
            "year": year,
            "group": gid,
            "features": ",".join(features),
            "R2_mean": df_fd["R2_val"].mean(),
            "R2_std":  df_fd["R2_val"].std(ddof=1),
            "RMSE_mean": df_fd["RMSE_val"].mean(),
            "RMSE_std":  df_fd["RMSE_val"].std(ddof=1),
            "MAE_mean":  df_fd["MAE_val"].mean(),
            "MAE_std":   df_fd["MAE_val"].std(ddof=1),
            "best_model_program": best_prog_str,
            "best_model_length":  best_prog_len,
            "best_model_depth":   best_prog_depth,
        }
        group_summary.append(summary_row)

        # Representative model record
        best_models_txt.append(
            f"[Year {year}][Group {gid}] features={features}\n"
            f"  Best (by val RMSE):\n"
            f"  program: {best_prog_str}\n"
            f"  length: {best_prog_len}  depth: {best_prog_depth}\n"
            f"  mean±sd R2:   {summary_row['R2_mean']:.3f} ± {summary_row['R2_std']:.3f}\n"
            f"  mean±sd RMSE: {summary_row['RMSE_mean']:.3f} ± {summary_row['RMSE_std']:.3f}\n"
            f"  mean±sd MAE:  {summary_row['MAE_mean']:.3f} ± {summary_row['MAE_std']:.3f}\n"
        )
        best_models_json.append({
            "year": year,
            "group": gid,
            "features": features,
            "best_program": best_prog_str,
            "best_length": best_prog_len,
            "best_depth": best_prog_depth,
            "cv_mean_std": {
                "R2":   [summary_row["R2_mean"], summary_row["R2_std"]],
                "RMSE": [summary_row["RMSE_mean"], summary_row["RMSE_std"]],
                "MAE":  [summary_row["MAE_mean"],  summary_row["MAE_std"]],
            }
        })

    # output detail file
    df_folds = pd.DataFrame(folds_detail)
    df_groups = pd.DataFrame(group_summary)

    folds_out = OUTDIR / f"cv_fold_metrics_detail_{year}.csv"
    groups_out = OUTDIR / f"cv_metrics_by_group_{year}.csv"
    best_txt = OUTDIR / f"best_models_{year}.txt"
    best_json = OUTDIR / f"best_models_{year}.json"

    df_folds.to_csv(folds_out, index=False, encoding="utf-8")
    df_groups.to_csv(groups_out, index=False, encoding="utf-8")
    with open(best_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(best_models_txt))
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best_models_json, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {groups_out}")
    print(f"[Saved] {folds_out}")
    print(f"[Saved] {best_txt}")
    print(f"[Saved] {best_json}")

def main():
    pd.set_option("display.width", 160)
    for y, cfg in INPUTS.items():
        run_year(y, cfg["data"], cfg["groups"])
    print("\n✅ GP + 5-fold CV complete")

if __name__ == "__main__":
    main()
