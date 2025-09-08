import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Optional: Set font sizes globally
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})
# Config
folders = {
    "results_NF3": 3,
    "results_NF5": 5,
    "results_NF7": 7
}
models = ["KNN", "XGBoost", "Random Forest", "KNN"]
seeds = [13, 42, 65, 789, 9999]
#colors = {3: '#D81B60', 5: '#1E88E5', 7: '#FFC107'}  
colors = ['#D81B60', '#1E88E5', '#FFC107']  # Colors for NF3, NF5, NF7

# Initialize data structures
mean_accuracies = {model: [] for model in models}
std_accuracies = {model: [] for model in models}
#
folder_list = list(folders.items())  # [(folder, feature_count), ...]
# Collect accuracies
for model in models:
    for folder,_ in folder_list:
        accs = []
        for seed in seeds:
            file_path = os.path.join(folder, f"{model}_confusion_matrix_{seed}.csv")
            if not os.path.exists(file_path):
                print(f"Missing: {file_path}")
                continue
            cm = pd.read_csv(file_path, header=0).values
            acc = np.trace(cm) / np.sum(cm)
            accs.append(acc)
        mean_accuracies[model].append(np.mean(accs))
        std_accuracies[model].append(np.std(accs))

# Plotting
x = np.arange(len(models))  # one group per model
width = 0.2  # width of each bar

fig, ax = plt.subplots(figsize=(10, 6))

# One bar per dataset per model
for i, (folder, label) in enumerate(folder_list):
    means = [mean_accuracies[model][i] for model in models]
    stds = [std_accuracies[model][i] for model in models]
    x_offset=x + (i - 1) * width
    ax.bar(x_offset, means, width, label=f"NF{label}", color=colors[i], yerr=stds, capsize=5)
    # Add value labels
    for xi, yi , si in zip(x_offset, means,stds):
        ax.text(xi, yi + si+0.01, f"{yi:.2f}", ha='center', va='bottom', fontsize=13)

# Formatting
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison Across Feature Sets")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(title="Feature Set",loc='upper center')
ax.set_ylim(0.5, 1.02)  
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# # Step 1: Load all 5 confusion matrices for KNN with 3 features
# KNN_folder = "results_NF3"
# KNN_conf_matrices = []

# for seed in seeds:
#     file_path = os.path.join(KNN_folder, f"KNN_confusion_matrix_{seed}.csv")
#     if not os.path.exists(file_path):
#         print(f"Missing: {file_path}")
#         continue
#     cm = pd.read_csv(file_path, header=0).values
#     KNN_conf_matrices.append(cm)

# # Step 2: Compute average confusion matrix
# if len(KNN_conf_matrices) > 0:
#     avg_cm = np.mean(KNN_conf_matrices, axis=0)
# else:
#     raise ValueError("No confusion matrices found for KNN with 3 features.")

# # Step 3: Plot the averaged confusion matrix
# fig, ax = plt.subplots(figsize=(6, 5))
# disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm)
# disp.plot(cmap="Blues", ax=ax, values_format=".2f")
# ax.set_title("Averaged Confusion Matrix (KNN, 3 Features)")
# plt.tight_layout()
# plt.show()


# Step 1: Load all 5 confusion matrices for KNN with 3 features

# Step 1: Load all 5 confusion matrices for KNN with 3 features
KNN_folder = "results_NF3"
KNN_conf_matrices = []

for seed in seeds:
    file_path = os.path.join(KNN_folder, f"KNN_confusion_matrix_{seed}.csv")
    if not os.path.exists(file_path):
        print(f"Missing: {file_path}")
        continue
    cm = pd.read_csv(file_path, header=0).values
    KNN_conf_matrices.append(cm)

# Step 2: Compute average confusion matrix
if len(KNN_conf_matrices) > 0:
    avg_cm = np.mean(KNN_conf_matrices, axis=0)
else:
    raise ValueError("No confusion matrices found for KNN with 3 features.")

# Step 3: Ensure diagonal represents correct predictions
if np.trace(avg_cm) < np.trace(avg_cm.T):
    avg_cm = avg_cm.T

# Step 4: Convert to percentages (row-wise normalization)
row_sums = avg_cm.sum(axis=1, keepdims=True)
cm_percent = (avg_cm / row_sums) * 100

# Step 5: Plot the averaged confusion matrix with bigger numbers
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_percent, interpolation="nearest", cmap="Blues")

# Add colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=12)

# Write numbers directly with bigger font
n_classes = cm_percent.shape[0]
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(
            j, i, f"{cm_percent[i, j]:.1f}",
            ha="center", va="center",
            color="white" if cm_percent[i, j] > 50 else "black",
            fontsize=14, fontweight="bold"
        )

# Labels and titles
ax.set_title("Averaged Confusion Matrix (%) - KNN, 3 Features", fontsize=16, weight="bold")
ax.set_xlabel("Predicted Label", fontsize=14)
ax.set_ylabel("True Label", fontsize=14)
ax.set_xticks(np.arange(n_classes))
ax.set_yticks(np.arange(n_classes))
ax.tick_params(axis="both", labelsize=12)

# Fix orientation: make y-axis go from 0 → n bottom-to-top
ax.invert_yaxis()

plt.tight_layout()
plt.show()
#
# ^___^
# \. ./
#  \o/
#