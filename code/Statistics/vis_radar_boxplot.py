import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi
from sklearn.preprocessing import MinMaxScaler
import os


colors = ['#CC6677', '#AA4499','#882255', "#4B3E8A"]  # CUD-friendly palette
level_names = ['Level 1', 'Level 2', 'Level 3', 'Level 4']


def create_radar(data, level_names, labels, angles,outdir):
    os.makedirs(outdir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (index, row) in enumerate(data.iterrows()):
        values = row.tolist()
        values += values[:1]  # complete the loop

        ax.plot(angles, values, label=level_names[i], color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Mean Feature Values Across Game Levels", y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f"{outdir}/radar10.png", dpi=300, bbox_inches='tight')
    plt.close()

def boxplot_per_feature(df, label_col='LEVEL', save=True, outdir='plots'):
      
    os.makedirs(outdir, exist_ok=True)

    # Get numeric feature columns only (explicitly excluding the label column)
    feature_cols = [col for col in df.columns if col != label_col and pd.api.types.is_numeric_dtype(df[col])]
    
    # Ensure the label column is treated as categorical
    df[label_col] = df[label_col].astype(str)

    # Get unique levels and create a fixed color mapping dictionary
    unique_levels = sorted(df[label_col].unique())
    color_mapping = {level: colors[i % len(colors)] for i, level in enumerate(unique_levels)}
    
     # Create the plots
    #sns.set(style='whitegrid')
    
    for feature in feature_cols:
        plt.figure(figsize=(6, 5))
        # Create a custom palette dictionary for seaborn to use
        custom_palette = {level: color_mapping[level] for level in unique_levels}
        # Create the boxplot with explicitly defined colors for each level
        ax = sns.boxplot(x=label_col, y=feature, data=df, order=unique_levels, palette=custom_palette)
        
        # Manually set the colors for each box
        for i, box in enumerate(ax.artists):
            box.set_facecolor(colors[i % len(colors)])
            box.set_edgecolor('black')
            
            # Also color the lines (whiskers, caps, etc)
            for j in range(i*6, i*6+6):
                if j < len(ax.lines):
                    ax.lines[j].set_color('black')
        
        plt.title(f"{feature} by Game Level", fontsize=15)
        plt.xlabel("Game Level", fontsize=14)
        plt.ylabel(feature, fontsize=14)
        plt.tick_params(axis='x', labelsize=14)  
        plt.tick_params(axis='y', labelsize=12)  
        plt.tight_layout()

        if save:
            plt.savefig(f"{outdir}/{feature}_boxplot.png", dpi=300, bbox_inches='tight')
            plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(9, 11), sharey=False)
    for j in range(len(features), len(axes.flat)):
        fig.delaxes(axes.flat[j])
    for i, feature in enumerate(features):
        ax = axes.flat[i]
        sns.boxplot(x=label_col, y=feature, data=df, order=unique_levels,
                    palette=custom_palette, ax=ax)
        ax.set_title(f'{i+1} - {feature}', fontsize=12)
        ax.set_xlabel("")  # remove to save space
        ax.set_ylabel("")
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=8)

    fig.suptitle("Feature distributions by game level", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
        
# Load your dataset
df = pd.read_excel(r'..\code\Statistics\selected_Dataset_7_stat_pervis.xlsx')

# Extract label and convert to categorical
label_col = 'LABEL'
df[label_col] = df[label_col].astype(str)
df = df.rename(columns={'$oo_{sd1/sd2}$': '$oo_{sd1sd2}$'})
# Drop SubjectID for the analysis
features = df.drop(columns=['SubjectID'])

# For the radar chart with normalized data
scaler = MinMaxScaler()
features_only = features.drop(columns=[label_col])
normalized_features = pd.DataFrame(
    scaler.fit_transform(features_only), 
    columns=features_only.columns
)

# Add the label back to the normalized data
normalized_data = pd.concat([normalized_features, df[label_col]], axis=1)

# Group by level and compute the mean of features for radar chart
mean_features = normalized_data.groupby(label_col).mean(numeric_only=True)

# Drop SubjectID if it's still present
if 'SubjectID' in mean_features.columns:
    mean_features = mean_features.drop(columns='SubjectID')

# Radar chart setup
labels = mean_features.columns
num_vars = len(labels)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # complete the loop

# Create the radar chart
create_radar(mean_features, level_names[:len(mean_features)], labels, angles, outdir='boxplots_radar')

# Create boxplots for all features
boxplot_per_feature(features, label_col=label_col, save=True, outdir='boxplots_radar')
#
# ^___^
# \. ./
#  \o/
#