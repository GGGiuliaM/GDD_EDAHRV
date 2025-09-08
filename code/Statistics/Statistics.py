import pandas as pd
from scipy.stats import shapiro
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

# NOTE that W stays both for W and ng2 and Q stays both for Q and F in th output file. Its a summarized version.

# Load your dataset
df = pd.read_excel("D:\\Git\\Neadex25\\data\\OfficialDatasetReduced2.xlsx")

# Rename and prepare columns
df.rename(columns={'SubjectID': 'subject_id', 'LABEL': 'level'}, inplace=True)
df['subject_id'] = df['subject_id'].astype(str)
df['level'] = df['level'].astype(str)

# Metadata columns to ignore
metadata_cols = ['LABEL_HL', 'Stress', 'Duration_Sec','$PSS_{Score}$', 'Performance']
features = [col for col in df.columns if col not in ['subject_id', 'level'] + metadata_cols]

results=[]

# Loop over each feature
for feature in features:
    print(f"\n{'='*60}\n Analyzing feature: {feature}")

    row = {'Feature': feature}
    # ------------------------------
    # 1. Normality check per level
    # ------------------------------
    normal = True
    print("Normality check (Shapiro-Wilk):")
    for lvl in sorted(df['level'].unique()):
        data = df[df['level'] == lvl][feature]
        stat, p = shapiro(data)
        row[f'Shapiro_p_Level_{lvl}'] = round(p, 4)
        print(f"  Level {lvl}: p = {p:.4f} {' normal' if p > 0.05 else ' non-normal'}")
        if p < 0.05:
            normal = False
    row['Is_Normal'] = normal
    # ------------------------------
    # 2. Statistical analysis
    # ------------------------------
    if normal:
        print("\n Running Repeated Measures ANOVA")
        try:
            row['Test'] = 'ANOVA'
            anova = pg.rm_anova(dv=feature, within='level', subject='subject_id', data=df, detailed=True)
            row['W'] = anova.loc[0, 'ng2'] if 'W' in anova.columns else None
            row['Q'] = anova.loc[0, 'F'] if 'F' in anova.columns else None
            row['ANOVA_p'] = anova.loc[0, 'p-unc']
            print(anova)

            posthoc = pg.pairwise_tests(dv=feature, within='level', subject='subject_id',
                                         data=df, padjust='bonf', parametric=True)
            print("\n Post-hoc pairwise t-tests (Bonferroni corrected):")
            available_cols = posthoc.columns
            cols_to_show = [col for col in ['A', 'B', 'T', 'W-val', 'p-unc', 'p-corr', 'sig', 'significant'] if col in available_cols]
            print(posthoc[cols_to_show])
            # Post-hoc summary: count and list significant comparisons
            sig_comparisons = posthoc[posthoc['p-corr'] < 0.05]
            row['PostHoc_Significant_Count'] = len(sig_comparisons)
            row['PostHoc_Significant_Pairs'] = ', '.join([f"{a}-{b}" for a, b in zip(sig_comparisons['A'], sig_comparisons['B'])])
    

        except Exception as e:
            print(f" Error in rmANOVA: {e}")


    else:
        print("\n Running Friedman Test (non-parametric alternative)")
        try:
            row['Test'] = 'Friedman'
            friedman = pg.friedman(dv=feature, within='level', subject='subject_id', data=df)
            print(friedman)
            type(friedman)
            row['W'] = friedman.loc['Friedman', 'W']
            row['Q'] = friedman.loc['Friedman','Q']
            row['ANOVA_p'] = friedman.loc['Friedman', 'p-unc']

            posthoc = pg.pairwise_tests(dv=feature, within='level', subject='subject_id',
                                         data=df, padjust='bonf', parametric=False) # NON parametric analysis
            print("\n Post-hoc Wilcoxon tests (Bonferroni corrected):")
            print("\nPost-hoc results:")
            print(posthoc[[col for col in ['A', 'B', 'T', 'W-val', 'p-unc', 'p-corr', 'sig','significant'] if col in posthoc.columns]])
            #print(posthoc[['A', 'B', 'W-val', 'p-unc', 'p-corr', 'sig']])

            # Post-hoc summary: count and list significant comparisons
            sig_comparisons = posthoc[posthoc['p-corr'] < 0.05]
            row['PostHoc_Significant_Count'] = len(sig_comparisons)
            row['PostHoc_Significant_Pairs'] = ', '.join([f"{a}-{b}" for a, b in zip(sig_comparisons['A'], sig_comparisons['B'])])
        except Exception as e:
            print(f" Error in Friedman test: {repr(e)}")
        
    # Append result
    results.append(row)   

    # ------------------------------
    # 3. Visualization
    # ------------------------------
    # plt.figure(figsize=(7, 5))
    # sns.boxplot(x='level', y=feature, data=df, palette='Set3')
    # sns.swarmplot(x='level', y=feature, data=df, color='0.25', alpha=0.6)
    # plt.title(f'{feature} across Game Levels')
    # plt.xlabel('Game Level')
    # plt.ylabel(feature)
    # plt.tight_layout()
    # plt.show()
    # plt.close()

results_df = pd.DataFrame(results)
results_df.to_excel("feature_statistics_summary.xlsx", index=False)
print("\n Results saved to 'feature_statistics_summary.xlsx'")