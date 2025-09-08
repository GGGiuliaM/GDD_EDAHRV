# GDD_EDAHRV
# Data and Code for Short-term Stress Intensity Detection during Exergaming

This repository contains the main datasets and scripts used in the study **“Short-term Stress Intensity Detection with Wearable Sensor during Exergaming”**. The work is a preliminary study on the feasibility of employing a lightweight, off-the-shelf, short-term (around 30s) stress detection system during the play of a custom exergame (Grab-Drag-Drop GDD). Twenty-three subjects completed four progressively stimuli-enriched game levels while their physiological signals (EDA, HRV) were recorded, after responding to to preliminary questionnaires. The analysis highlights the extent of arousal detection and characterizes the patterns of the detected responses. In addition to physiological measures, questionnaire-derived information and in-game performance metrics are considered as potential influencing factors. The primary objective of this study is to evaluate whether the proposed setup can reliably identify different levels of induced stress in a realistic scenario, with the long-term aim of supporting adaptive rehabilitative exergaming. A secondary objective is to identify the most informative features for stress detection, enabling optimization of the setup for future developments. Finally, secondary outcomes and potenti  

The provided code performs statistical analyses and classification to detect short-term stress responses and discriminate game levels based on arousal. These resources support the findings presented in the paper, which demonstrate the feasibility of using lightweight wearable systems for monitoring stress during interactive gameplay.  

For more details, refer to the [paper](insert-link-to-your-paper-if-available).  

---

## Data

| File | Description | Used by |
|------|-------------|---------|
| [Dataset_Original_Completo.xlsx](data/Dataset_Original_Completo.xlsx) | Complete original dataset with all computed features and data points. | General reference |
| [Dataset_Original_Reduced_forMM.xlsx](data/Dataset_Original_Reduced_forMM.xlsx) | Dataset for Linear Mixed Models, reduced troughout feature selection. For each feature cluster, the feature most correlated with **performance** is selected. | `scripts/Mixedmodels.py` |
| [game_meanPerf.xlsx](data/game_meanPerf.xlsx) | Mean performance values for each game. | Performance summary analysis |
| [game_mean_PSS.xlsx](data/game_mean_PSS.xlsx) | Features averaged across levels. Features are selected using top correlation with Perceived Stress Score (PSS). | Baseline Stress related analysis |
| [H_data_original.csv](data/H_data_original.csv) | Features as obtained by the EDA analysis. | General reference |
| [Dataset_Original_Reduced_levels.xlsx](data/Dataset_Original_Reduced_levels.xlsx) | Official reduced dataset for **level analysis**, based on clusters. Picks the feature most correlated with level label. | `code/statistic.py` |
| [selected_Dataset_3_stat.xlsx](data/selected_Dataset_3_stat.xlsx) | Dataset subset with 3 selected features for statistical analysis. | Statistical analysis |
| [selected_Dataset_5_stat.xlsx](data/selected_Dataset_5_stat.xlsx) | Dataset subset with 5 selected features for statistical analysis. | Statistical analysis |
| [selected_Dataset_7_stat.xlsx](data/selected_Dataset_7_stat.xlsx) | Dataset subset with 7 selected features for statistical analysis. | Statistical analysis |

---

## Scripts

| Script | Description | Uses Dataset |
|--------|------------|-------------|
| [Mixedmodels.py](code/Mixedmodels.py) | Performs Linear Mixed Model analysis to evaluate performance-related features. | `Dataset_Original_Reduced_forMM.xlsx` |
| [statistic.py](code/statistic.py) | Performs statistical analysis on level-related features based on clusters. | `Dataset_Original_Reduced_levels.xlsx` |

---

## Repository Structure

