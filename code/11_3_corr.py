from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


BASE   = Path(r"D:\PPT_Word_Exel\2025\Final_Project\WorkSpace2\model")
QC_DIR = BASE / "QC"
OUTDIR = BASE / "Corr"
OUTDIR.mkdir(parents=True, exist_ok=True)

FILES = {
    2019: QC_DIR / "merged_2019_qc_final.csv",
    2022: QC_DIR / "merged_2022_qc_final.csv",
}

# ---------- Columns & labels (match the paper's order) ----------
COLS_ORDER = [
    "temperature",     # Air temperature (station)
    "soil_moisture",   # Soil moisture (ERA5-Land layer1)
    "emis31",          # Emissivity (MODIS band31)
    "rain",            # Rainfall (COMEPHORE, mm/h)
    "ndvi",            # NDVI
    "mndwi",           # MNDWI
    "t2m",             # 2-m temperature (ERA5-Land)
    "elevation",       # Elevation (SRTM)
]

LABELS = [
    "Air temperature\n(station)",
    "Soil moisture",
    "Emissivity",
    "Rainfall",
    "NDVI",
    "MNDWI",
    "2-m temperature\n(ERA5-Land)",
    "Elevation",
]

# ---------- helper: triangular heatmap ----------
def plot_lower_triangle(corr_df: pd.DataFrame, labels, out_png: Path, title=None,
                        vmin=-0.8, vmax=1.0, annotate=True):
    """
    Draw a lower-triangular correlation heatmap with red colormap (paper-like).
    """
    C = corr_df.values.astype(float)
    n = C.shape[0]

    mask = np.triu(np.ones_like(C, dtype=bool), k=1)
    M = np.ma.array(C, mask=mask)

    # The more positive the correlation, the redder it is.
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(8.2, 7.0), dpi=220)  # high dpi
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    # mark r value
    if annotate:
        for i in range(n):
            for j in range(n):
                if i >= j:
                    val = C[i, j]
                    txt = f"{val:.3f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color="black")


    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)


    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("r", rotation=0, labelpad=8)
    cbar.ax.tick_params(labelsize=8)

    if title:
        ax.set_title(title, fontsize=10, pad=10)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def run_one(year: int, csv_path: Path):
    print(f"== Year {year} ==")
    if not csv_path.exists():
        print(f"[Skip] Not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    keep_cols = [c for c in COLS_ORDER if c in df.columns]
    use = df[keep_cols].dropna(how="any").copy()

    # Calculate Pearson correlation
    corr = use.corr(method="pearson")

    corr = corr.loc[COLS_ORDER, COLS_ORDER]

    out_csv = OUTDIR / f"corr_matrix_{year}.csv"
    corr.to_csv(out_csv, encoding="utf-8")
    print(f"[Saved] {out_csv}")


    out_png = OUTDIR / f"corr_triangular_{year}.png"
    plot_lower_triangle(corr, LABELS, out_png,
                        title=f"({ 'a' if year==2019 else 'b' })  {year}",
                        vmin=-0.8, vmax=1.0, annotate=True)
    print(f"[Saved] {out_png}")

def main():
    mpl.rcParams["axes.edgecolor"] = "0.6"
    mpl.rcParams["axes.linewidth"] = 0.6

    for y, fp in FILES.items():
        run_one(y, fp)

    print(f"\n✅ All done. See outputs in: {OUTDIR}")

if __name__ == "__main__":
    main()
