from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


BASE = Path(r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\model")
QC_DIR = BASE / "QC"
OUTDIR = BASE / "Stepwise"
OUTDIR.mkdir(parents=True, exist_ok=True)

FILES = {
    2019: QC_DIR / "merged_2019_qc_final.csv",
    2022: QC_DIR / "merged_2022_qc_final.csv",
}

# ---------------- Variables ----------------
TARGET = "temperature"
FORCED = ["t2m"]  # t2m we always need
# in the next will order
LOCAL_VARS = ["emis31","ndvi","mndwi","elevation","soil_moisture","rain"]
ALL_VARS = FORCED + LOCAL_VARS


#Here std is the population standard deviation
def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0)

def fit_ols_z(y: pd.Series, X: pd.DataFrame):

 #z-score space fitting, then returning the model and common indicators
    Xz = zscore(X.copy())
    yz = zscore(y.copy())
    Xz = sm.add_constant(Xz, has_constant='add')
    model = sm.OLS(yz, Xz, missing='drop').fit()
    aic = model.aic
    r2 = model.rsquared
    r2a = model.rsquared_adj
    return model, aic, r2, r2a

def forward_stepwise(df: pd.DataFrame, forced=FORCED, candidates=LOCAL_VARS, p_thresh=0.05):
    """
    fore t2m
    """
    y = df[TARGET].astype(float)
    selected = list(forced)
    remaining = [v for v in candidates if v not in selected]
    logs = []

    # 初始模型（只有 forced）
    m0, aic0, r20, r2a0 = fit_ols_z(y, df[selected])
    logs.append({"step": 0, "added": "(init)", "k": len(selected), "AIC": aic0, "R2_adj": r2a0})

    step = 0
    while remaining:
        step += 1
        best_var = None
        best_p = None
        best_model = None
        best_aic = None
        best_r2a = None

        # Try adding each remaining variable one by one
        for var in remaining:
            X_try = df[selected + [var]].astype(float)
            model, aic, r2, r2a = fit_ols_z(y, X_try)
            p_new = model.pvalues.get(var, np.nan)
            if np.isnan(p_new):
                continue
            if (best_p is None) or (p_new < best_p):
                best_p = p_new
                best_var = var
                best_model = model
                best_aic = aic
                best_r2a = r2a

        if best_var is None or np.isnan(best_p):
            break


        if best_p <= p_thresh:
            selected.append(best_var)
            remaining.remove(best_var)
            logs.append({"step": step, "added": best_var, "k": len(selected), "p_added": float(best_p),
                         "AIC": float(best_aic), "R2_adj": float(best_r2a)})
        else:
            #no var top
            break

    log_df = pd.DataFrame(logs)
    return selected, log_df

def beta_ranking_fullmodel(df: pd.DataFrame, all_vars=ALL_VARS):
    """
    use beta to order
    """
    y = df[TARGET].astype(float)
    X = df[all_vars].astype(float)
    model, aic, r2, r2a = fit_ols_z(y, X)

    params = model.params.drop("const", errors="ignore")
    tvals = model.tvalues.drop("const", errors="ignore")
    pvals = model.pvalues.drop("const", errors="ignore")
    out = pd.DataFrame({
        "variable": params.index,
        "beta": params.values,
        "t": tvals.values,
        "p": pvals.values
    })
    out["abs_beta"] = out["beta"].abs()
    out = out.sort_values("abs_beta", ascending=False).reset_index(drop=True)

    # rank
    local_beta = out[out["variable"].isin(LOCAL_VARS)].copy()
    local_beta = local_beta.sort_values("abs_beta", ascending=False).reset_index(drop=True)
    return out, local_beta

def build_groups_from_local_ranking(local_order):
    """
    from bete to group
    """
    order = list(local_order)
    groups = {}
    for k in range(1, 7):
        groups[k] = ["t2m"] + order[:k]
    return groups

# ---------------- Main ----------------
def main():
    pd.set_option("display.width", 160)
    for year, fp in FILES.items():
        print(f"\n=== Year {year} ===")
        if not fp.exists():
            print(f"[Skip] Not found: {fp}")
            continue

        df_raw = pd.read_csv(fp)

        keep = [TARGET] + ALL_VARS
        df = df_raw[keep].dropna(how="any").reset_index(drop=True)

        #move forwrad to add value #
        selected, log_df = forward_stepwise(df, forced=FORCED, candidates=LOCAL_VARS, p_thresh=0.05)
        out_log = OUTDIR / f"stepwise_sequence_{year}.csv"
        log_df.to_csv(out_log, index=False, encoding="utf-8")
        print(f"[Saved] {out_log}")
        print(" Selected (order):", " → ".join(selected))

        # —— standered Beta —— #
        beta_all, beta_local = beta_ranking_fullmodel(df, all_vars=ALL_VARS)
        out_beta = OUTDIR / f"beta_fullmodel_{year}.csv"
        beta_all.to_csv(out_beta, index=False, encoding="utf-8")
        print(f"[Saved] {out_beta}")

        # from beta to 6 group —— #
        local_order = beta_local["variable"].tolist()
        groups = build_groups_from_local_ranking(local_order)
        out_groups = OUTDIR / f"groups_from_beta_{year}.json"
        with open(out_groups, "w", encoding="utf-8") as f:
            json.dump({"year": year, "local_order_by_abs_beta": local_order, "groups": groups}, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {out_groups}")
        print(" Local order by |Beta|:", local_order)
        print(" Groups preview:", groups[1], " / ", groups[6])

    print("\n✅ Stepwise + Beta complete。next step to di six GP + 5-fold CV。")

if __name__ == "__main__":
    main()
