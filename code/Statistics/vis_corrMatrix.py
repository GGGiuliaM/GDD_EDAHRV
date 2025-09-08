# Mi sono brasata il codice non pullando e non committando dall'altro pc :(

# CORRELATION MATRIX TOP features, ranked dalla statistica nel file
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

# === Load CSV ===
file_path = "selected_Dataset_7_stat.xlsx"


df = pd.read_excel(file_path)


# Drop subjectID if it exists
if "subjectID" in df.columns:
  df = df.drop(columns=["SubjectID"])


# === Compute Correlations ===
pearson_corr = df.corr(method="pearson").abs()
spearman_corr = df.corr(method="spearman").abs()


# === Plotting Function ===
def plot_corr_heatmap(corr_matrix, title):
  plt.figure(figsize=(10, 8))
  # "coolwarm" works for colorblind + prints well in grayscale
  sns.heatmap(
    corr_matrix,
    cmap="PuRd",
    annot=True,
    fmt=".2f",
    cbar=True,
    square=True,
    annot_kws={"size": 12}, # annotation font size
  )
  plt.title(title, fontsize=16, pad=20)
  plt.xticks(rotation=45, ha="right", fontsize=12)
  plt.yticks(fontsize=12)
  plt.tight_layout()
  plt.show()


# === Plot both ===
#plot_corr_heatmap(pearson_corr, "Absolute Pearson Correlation")
plot_corr_heatmap(spearman_corr, "Absolute Spearman Correlation")
