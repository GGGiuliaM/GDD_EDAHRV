import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import numpy as np
from statsmodels.stats.multitest import multipletests 

# LOAD DATA
df = pd.read_excel("..\Dataset_original_Reduced_forMM.xlsx")

# Prepare columns
df['SubjectID'] = df['SubjectID'].astype(str)
df['LABEL'] = df['LABEL'].astype('category')

metadata_cols = ['$PSS_{Score}$', 'LABEL_HL', 'Stress', 'Duration_Sec', 'SubjectID', 'LABEL', 'Performance','Score']
features = [col for col in df.columns if col not in metadata_cols]

results = []
num_feat = len(features) 

for feature in features:
    print(f"Analyzing feature: {feature}")
    formula = f'Q("{feature}") ~ LABEL + Performance' # Metti + per non interazione, metti * per includere interazione

    try:
        model = mixedlm(formula, df, groups=df["SubjectID"], re_formula="~1")
        result = model.fit()

        label_coeffs = {param_name: param_value for param_name, param_value in result.params.items() if param_name.startswith('LABEL[T.')}
        p_perf_unc = result.pvalues.get('Performance', None)
        
        # Store for later Bonferroni & FDR calculation
        results.append({
            'feature': feature,
            'p_performance_uncorrected': p_perf_unc,
            'coef_performance': result.params.get('Performance', None),
            'label_coefficients': label_coeffs,
            'full_summary': result.summary().as_text()
        })

    except Exception as e:
        print(f"Error fitting model for feature {feature}: {e}")
        results.append({
            'feature': feature,
            'p_performance_uncorrected': None,
            'coef_performance': None,
            'label_coefficients': {},
            'full_summary': str(e)
        })

results_df = pd.DataFrame(results)

# --- Apply Multiple Comparisons Corrections ---

# 1. Get all uncorrected p-values for Performance from successful models
#    We only include p-values that are not None (i.e., model converged)
valid_p_values_series = results_df['p_performance_uncorrected'].dropna()
original_features_with_valid_p = valid_p_values_series.index # Store original index to map back

if not valid_p_values_series.empty:
    # Convert to list for multipletests function
    p_values_list = valid_p_values_series.tolist()
    
    # Bonferroni Correction 
    _, p_values_bonferroni_corrected, _, _ = multipletests(p_values_list, alpha=0.05, method='bonferroni')
    
    # Benjamini-Hochberg FDR Correction
    _, p_values_fdr_corrected, _, _ = multipletests(p_values_list, alpha=0.05, method='fdr_bh')

    # Map corrected p-values back to the DataFrame
    results_df.loc[original_features_with_valid_p, 'p_performance_bonferroni'] = p_values_bonferroni_corrected
    results_df.loc[original_features_with_valid_p, 'p_performance_fdr_corrected'] = p_values_fdr_corrected
else:
    results_df['p_performance_bonferroni'] = np.nan
    results_df['p_performance_fdr_corrected'] = np.nan

# --- Generate the Summary Report ---

# Filter for features where Performance effect is significant (uncorrected p < 0.05)
significant_performance_features_df = results_df[
    (results_df['p_performance_uncorrected'] < 0.05) &
    (results_df['p_performance_uncorrected'].notna())
].copy() 

# Expand LABEL coefficients into separate columns for easier CSV export
if not significant_performance_features_df.empty:
    all_label_param_names = set()
    for d in significant_performance_features_df['label_coefficients']:
        all_label_param_names.update(d.keys())
    
    for label_col_name in sorted(list(all_label_param_names)):
        significant_performance_features_df[label_col_name] = significant_performance_features_df['label_coefficients'].apply(
            lambda x: x.get(label_col_name, np.nan)
        )
    
    significant_performance_features_df = significant_performance_features_df.drop(
        columns=['label_coefficients', 'full_summary']
    )

    report_cols_order = [
        'feature', 
        'p_performance_uncorrected', 
        'p_performance_bonferroni', 
        'p_performance_fdr_corrected', 
        'coef_performance'
    ] + sorted(list(all_label_param_names))

    significant_performance_features_df = significant_performance_features_df[report_cols_order]

    report_filename = "significant_performance_features_report.csv"
    significant_performance_features_df.to_csv(report_filename, index=False)
    
    print(f"\n--- Report Generated ---")
    print(f"Report saved to {report_filename}")
    print(f"Contains {len(significant_performance_features_df)} features with uncorrected p-value < 0.05 for 'Performance'.")
    print("\nSummary of significant features (first 5 rows):")
    print(significant_performance_features_df.head())

else:
    print("\nNo features found with a statistically significant performance effect (uncorrected p < 0.05).")

#
# ^___^
# \. ./
#  \o/
#