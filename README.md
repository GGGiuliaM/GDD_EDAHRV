# GDD_EDAHRV
# Data and Code for Short-term Stress Intensity Detection during Exergaming

This repository contains the main datasets and scripts used in the study **“Short-term Stress Intensity Detection with Wearable Sensor during Exergaming”**. The work is a preliminary study on the feasibility of employing a lightweight, off-the-shelf, short-term (around 30s) stress detection system during the play of a custom exergame (Grab-Drag-Drop GDD). Twenty-three subjects completed four progressively stimuli-enriched game levels while their physiological signals (EDA, BVP, throughout Empatica E4) were recorded, after responding to to preliminary questionnaires. The analysis highlights the extent of arousal detection and characterizes the patterns of the detected responses. In addition to physiological measures, questionnaire-derived information and in-game performance metrics are considered as potential influencing factors. The primary objective of this study is to evaluate whether the proposed setup can reliably identify different levels of induced stress in a realistic scenario, with the long-term aim of supporting adaptive rehabilitative exergaming. A secondary objective is to identify the most informative features for stress detection, enabling optimization of the setup for future developments. Finally, secondary outcomes and potenti  

The provided code performs statistical analyses and classification to detect short-term stress responses and discriminate game levels based on arousal. These resources support the findings presented in the paper, which demonstrate the feasibility of using lightweight wearable systems for monitoring stress during interactive gameplay.  

For more details, refer to the [paper](insert-link-to-your-paper-if-available) NOTE: STILL UNDER PEER-REVIEW.  

---

## Data

| File | Description | Used by |
|------|-------------|---------|
| [Dataset_Original_Completo.xlsx](data/Dataset_Original_Completo.xlsx) | Complete original dataset with all computed features and data points. | General reference |
| [Dataset_Original_Reduced_forMM.xlsx](data/Dataset_Original_Reduced_forMM.xlsx) | Dataset for Linear Mixed Models, reduced troughout feature selection. For each feature cluster, the feature most correlated with **performance** is selected. | Influencing factors, `scripts/Mixedmodels.py` |
| [game_meanPerf.xlsx](data/game_meanPerf.xlsx) | Features averaged across levels. Features are selected using top correlation with **performance**.  | Jamovi - Performance correlation study |
| [game_mean_PSS.xlsx](data/game_mean_PSS.xlsx) | Features averaged across levels. Features are selected using top correlation with **Perceived Stress Score (PSS)**. | Jamovi- Baseline Stress related correlation study |
| [H_data_original.csv](data/H_data_original.csv) | Features as obtained by the EDA analysis. | General reference, used to complete: 'Dataset_Original_Completo.xlsx'  |
| [Dataset_Original_Reduced_levels.xlsx](data/Dataset_Original_Reduced_levels.xlsx) | Official reduced dataset for **level** analysis, based on clusters. Picks the feature most correlated with level label. | Statistical Analysis `code/statistic.py` |
| [selected_Dataset_3_stat.xlsx](data/selected_Dataset_3_stat.xlsx) | Dataset subset with top 3 manually selected features. | ML |
| [selected_Dataset_5_stat.xlsx](data/selected_Dataset_5_stat.xlsx) | Dataset subset with top 5 manually selected features. | ML |
| [selected_Dataset_7_stat.xlsx](data/selected_Dataset_7_stat.xlsx) | Dataset subset with top 7 manually selected features. | ML |

---

## Scripts

| Script | Description | Uses Dataset |
|--------|------------|-------------|
| [statistic.py](code/Statistics/statistic.py) | Performs ANOVA/FRIEDMAN statistical analysis on features for the level analysis. | `Dataset_Original_Reduced_levels.xlsx` |
| [vis_corrMatrix.py](code/Statistics/vis_corrMatrix.py) | Script to compute and visualize the correlation matrix | `selected_Dataset_7_stat.xlsx` |
| [vis_radar_boxplot.py](code/Statistics/vis_radar_boxplot.py) | Script to visualize boxplots and radar plot| `selected_Dataset_7_stat.xlsx` |
| [main_ML_loso.py](code/ML/main_ML_loso.py) | Performs level classification with LOSO cross validation. Models: KNN, SVM, RF, XGBoost( with SHAP). 5 seeds repetition.| `selected_Dataset_7_stat.xlsx`,  `selected_Dataset_5_stat.xlsx`,  `selected_Dataset_3_stat.xlsx` |
| [vis_accuracy_CM.py](code/ML/vis_accuracy_CM.py) | script to visualize the accuracy comparison plot across seeds and the best model confusion matrix | `selected_Dataset_7_stat.xlsx`,  `selected_Dataset_5_stat.xlsx`,  `selected_Dataset_3_stat.xlsx` |
| [vis_featImportance.py](code/ML/vis_featImportance.py) | script to visualize the ranking obtained from SHAP values across 5 seeds| `selected_Dataset_7_stat.xlsx`,  `selected_Dataset_5_stat.xlsx`,  `selected_Dataset_3_stat.xlsx` |
| [Mixedmodels.py](code/Influencingfactors/Mixedmodels.py) | Performs Linear Mixed Model analysis to evaluate performance effects on features. | `Dataset_Original_Reduced_forMM.xlsx` |

## Results
| File/Folder | Description | Generated by |
| [feature_statistic_summary.xlsx](code/statistics/feature_statistic_summary.xlsx) | Results of the statistical Analysis between levels., `statistic.py` |
| [results_NF3](code/ML/results_NF3) | Results obtained from the ML pipeline for features set = 3 | `main_ML_loso.py` |
| [results_NF5](code/ML/results_NF5) | Results obtained from the ML pipeline for features set = 5 | `main_ML_loso.py`  |
| [results_NF7](code/ML/results_NF7) | Results obtained from the ML pipeline for features set = 7 | `main_ML_loso.py`  |
| [significant_performance_features_report.csv](code/Influencingfactors/significant_performance_features_report.csv) | Results of the Linear Mixed Models, that analizes the effect of performance. | Influencing factors, `Mixedmodels.py` |





