import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from pathlib import Path


# ====== Config ======
PATTERN = r'..\ML\results_NF7\feature*.csv'  
TOP_K = None                           # e.g., 25 to keep top-k features by mean importance
OUTPUT = "shap_importance_beeswarm.png"
SAVE_SVG = True
RANDOM_STATE = 42
JITTER = 0.12                          # vertical jitter for seed points
POINT_SIZE = 20
MEAN_SIZE = 60

# ====== Load & align ======
files = sorted(glob.glob(PATTERN))
if not files:
    raise FileNotFoundError(f"No CSV files matched pattern: {PATTERN}")

series_list = []
colnames = []
for f in files:
    df = pd.read_csv(f)
    # Standardize column names and types
    df.columns = [str(c).strip() for c in df.columns]
    if not {"Feature", "Importance"}.issubset(df.columns):
        raise ValueError(f"{f} must have columns 'Feature' and 'Importance'")

    # Clean and aggregate duplicates within a file (if any)
    s = (
        df[["Feature", "Importance"]]
        .dropna(subset=["Feature", "Importance"])  # keep valid rows
        .assign(Feature=lambda d: d["Feature"].astype(str).str.strip())
        .groupby("Feature", as_index=True)["Importance"].mean()
    )
    series_list.append(s)
    colnames.append(Path(f).stem)

# Outer-join by feature name to align across seeds regardless of order
merged = pd.concat(series_list, axis=1)
merged.columns = colnames

# ====== Stats across seeds ======
means = merged.mean(axis=1, skipna=True)
stds  = merged.std(axis=1, ddof=1, skipna=True)
counts = merged.count(axis=1)

# Order features by mean importance (desc)
order = means.sort_values(ascending=False).index
merged = merged.loc[order]
means  = means.loc[order]
stds   = stds.loc[order]
counts = counts.loc[order]

# Optionally keep only top-k features
if TOP_K is not None:
    merged = merged.iloc[:TOP_K]
    means  = means.iloc[:TOP_K]
    stds   = stds.iloc[:TOP_K]
    counts = counts.iloc[:TOP_K]

# ====== Beeswarm plot ======
rng = np.random.default_rng(RANDOM_STATE)
plt.figure(figsize=(7,5))
y_positions = np.arange(len(merged))

for i, feat in enumerate(merged.index):
    vals = merged.loc[feat].dropna().values
    if len(vals) == 0:
        continue
    jitter = rng.uniform(-JITTER, JITTER, size=len(vals))
    y = i + jitter
    # individual seed points
    plt.scatter(vals, y, s=POINT_SIZE, alpha=0.5, linewidths=0.5, color='blue', edgecolors="white",)
    # mean marker
    plt.scatter(means[feat], i, s=MEAN_SIZE, marker="D", zorder=3, label="Mean" if i == 0 else None,color='purple')
    # std as horizontal error bar (only if we have >1 seed for this feature)
    if np.isfinite(stds[feat]) and counts[feat] > 1:
        plt.errorbar(means[feat], i, xerr=stds[feat], fmt="none", capsize=4, linewidth=1,color='pink')

plt.yticks(y_positions, merged.index)
plt.xlabel("SHAP importance (per-seed aggregation)")
plt.title("Feature importances across seeds")
plt.legend(loc="lower right")
plt.tick_params(axis="both", labelsize=13)
plt.gca().invert_yaxis()  # put most important at the top
plt.tight_layout()

# Save outputs
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
if SAVE_SVG:
    plt.savefig(Path(OUTPUT).with_suffix(".svg"), bbox_inches="tight")
plt.show()

# Also export a tidy summary table
summary = pd.DataFrame({
    "mean": means,
    "std": stds,
    "n_seeds": counts
})
summary.to_csv("shap_importance_summary.csv")
#
# ^___^
# \. ./
#  \o/
#