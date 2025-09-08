import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, accuracy_score
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

import shap
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import warnings
import os
warnings.filterwarnings('ignore')

# Load the dataset
def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        print(f"Dataset loaded successfully with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Data preprocessing function
def preprocess_data(data):
    required_cols = ['SubjectID', 'LABEL']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset")
    X = data.drop(['SubjectID', 'LABEL'], axis=1)
    y = data['LABEL']
    subject_ids = data['SubjectID']
    print(f"Class distribution:\n{y.value_counts().sort_index()}")
    print(f"Number of unique subjects: {len(subject_ids.unique())}")
    feature_names = X.columns.tolist()
    return X, y, subject_ids, feature_names

# Define search spaces
def get_search_spaces():
    svm_space = {
        'C': Real(1e-3, 1e3, prior='log-uniform'),
        'gamma': Real(1e-6, 1e1, prior='log-uniform'),
        'kernel': Categorical(['rbf', 'linear', 'poly', 'sigmoid']),
        'degree': Integer(1, 5),
    }
    rf_space = {
        'n_estimators': Integer(10, 300),
        'max_depth': Integer(2, 7),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 20),
        'max_features': Categorical(['sqrt', 'log2', None]),
    }
    knn_space = {
        'n_neighbors': Integer(1, 9),
        'weights': Categorical(['uniform', 'distance']),
        'metric': Categorical(['euclidean', 'manhattan', 'minkowski']),
        'p': Integer(3, 5)
    }
    xgb_space = {
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(2, 7),
        'min_child_weight': Integer(1, 10),
        'gamma': Real(0, 1, prior='uniform'),
        'subsample': Real(0.5, 1.0, prior='uniform'),
        'colsample_bytree': Real(0.5, 1.0, prior='uniform'),
    }
    
    return svm_space, rf_space, knn_space, xgb_space


# Create models
def create_models(search_spaces, RANDOM_SEED=42):
    svm_space, rf_space, knn_space, xgb_space = search_spaces

    xgb_model = BayesSearchCV(
        xgb.XGBClassifier(
            objective='multi:softprob', eval_metric='mlogloss', random_state=RANDOM_SEED
        ),
        xgb_space, n_iter=20, cv=3, n_jobs=-1, random_state=RANDOM_SEED
    )
    
    svm_model = BayesSearchCV(
        SVC(probability=True, random_state=RANDOM_SEED),
        svm_space, n_iter=20, cv=3, n_jobs=-1, random_state=RANDOM_SEED
    )
    rf_model = BayesSearchCV(
        RandomForestClassifier(random_state=RANDOM_SEED),
        rf_space, n_iter=20, cv=3, n_jobs=-1, random_state=RANDOM_SEED
    )
    knn_model = BayesSearchCV(
        KNeighborsClassifier(),
        knn_space, n_iter=20, cv=3, n_jobs=-1, random_state=RANDOM_SEED
    )

    return {
        'XGBoost': xgb_model,
        'SVM': svm_model,
        'Random Forest': rf_model,
        'KNN': knn_model,
      
    }

# Evaluate models with LOSO
def evaluate_models_loso(models, X, y, subject_ids, feature_names):
    logo = LeaveOneGroupOut()
    results = {}
    needs_scaling = ['SVM', 'KNN']
    xgb_feature_importances = []

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        all_y_true, all_y_pred, all_y_proba = [], [], []
        subject_performances = {}

        for train_idx, test_idx in logo.split(X, y, subject_ids):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            test_subject = subject_ids.iloc[test_idx].iloc[0]

            if model_name in needs_scaling:
                scaler = StandardScaler()
                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

            if model_name == 'XGBoost':
                y_train_xgb = y_train - 1
                model.fit(X_train, y_train_xgb)
                xgb_estimator = model.best_estimator_

                explainer = shap.TreeExplainer(xgb_estimator)
                shap_values = explainer.shap_values(X_train) #(n_samples, n_features, n_classes)

                print(f"Type of shap_values: {type(shap_values)}")

                #if isinstance(shap_values, list):
                    #for i, sv in enumerate(shap_values):
                      #  print(f"Class {i}: shap_values[{i}].shape = {sv.shape}")
                   # shap_array = np.stack(shap_values)  # shape: (num_classes, num_samples, num_features)
                  #  print(f"Stacked shap_values shape: {shap_array.shape}")
               # else:
               #     print(f"shap_values.shape = {shap_values.shape}")


                # Average SHAP values: first over samples, then over classes
                shap_values_mean = np.mean(np.abs(shap_values), axis=1)  # shape: (num_classes, num_features)
                mean_shap_importance = np.mean(np.abs(shap_values), axis=(0, 2))  

                assert mean_shap_importance.shape[0] == len(feature_names), "Mismatch between SHAP features and actual features"
                
                shap_series = pd.Series(mean_shap_importance, index=feature_names)
                xgb_feature_importances.append(shap_series)
                
                y_pred_xgb = model.predict(X_test)
                y_pred = y_pred_xgb + 1
                y_proba = model.predict_proba(X_test)
          
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_proba)

            # Per-subject metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            if len(np.unique(y_test)) > 1:
                roc_auc = roc_auc_score(
                    pd.get_dummies(y_test),
                    y_proba,
                    multi_class='ovr',
                    average='weighted'
                )
            else:
                roc_auc = np.nan

            subject_performances[test_subject] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'roc_auc': roc_auc
            }
            print(f"  Subject {test_subject}: F1 = {f1:.4f}, Acc = {accuracy:.4f}")

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.array(all_y_proba)

        overall_accuracy = accuracy_score(all_y_true, all_y_pred)
        precision = precision_score(all_y_true, all_y_pred, average='weighted')
        recall = recall_score(all_y_true, all_y_pred, average='weighted')
        f1 = f1_score(all_y_true, all_y_pred, average='weighted')
        roc_auc = roc_auc_score(
            pd.get_dummies(all_y_true),
            all_y_proba,
            multi_class='ovr',
            average='weighted'
        )
        cm = confusion_matrix(all_y_true, all_y_pred)
        best_params = getattr(model, 'best_params_', {}) 
        
        results[model_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': overall_accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_true': all_y_true,
            'y_pred': all_y_pred,
            'y_proba': all_y_proba,
            'subject_performances': subject_performances,
            'best_params': model.best_params_
        }

        print(f"\n{model_name} Results:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {overall_accuracy:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  Best Parameters: {model.best_params_}")

    if xgb_feature_importances:
        shap_df = pd.DataFrame(xgb_feature_importances).T
        mean_importances = shap_df.mean(axis=1)
        results['feature_importance'] = mean_importances.sort_values(ascending=False)

    return results

# Save results
def visualize_results(results, output_dir='results', SEED=42):
    os.makedirs(output_dir, exist_ok=True)
    models = [m for m in results.keys() if m != 'feature_importance']
    metrics = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']

    # Overall performance
    performance_data = []
    for model in models:
        model_metrics = {metric: results[model][metric] for metric in metrics}
        model_metrics['model'] = model
        performance_data.append(model_metrics)
    pd.DataFrame(performance_data).to_csv(os.path.join(output_dir, f'performance_metrics_{SEED}.csv'), index=False)

    # Confusion matrices
    for model in models:
        pd.DataFrame(results[model]['confusion_matrix']).to_csv(
            os.path.join(output_dir, f'{model}_confusion_matrix_{SEED}.csv'), index=False
        )

    # Subject-level performance
    for model in models:
        subject_df = pd.DataFrame.from_dict(results[model]['subject_performances'], orient='index').reset_index()
        subject_df.rename(columns={'index': 'subject_id'}, inplace=True)
        subject_df.to_csv(os.path.join(output_dir, f'{model}_subject_performance_{SEED}.csv'), index=False)

    # Feature importance CSV
    if 'feature_importance' in results:
        pd.DataFrame({
            'Feature': results['feature_importance'].index,
            'Importance': results['feature_importance'].values
        }).sort_values('Importance', ascending=False).to_csv(
            os.path.join(output_dir, f'feature_importance_{SEED}.csv'), index=False
        )

# Main
def main():
    seeds = [13, 42, 65, 789, 9999]
    file_paths = [
        "selected_Dataset_7_stat.xlsx", 
        "selected_Dataset_5_stat.xlsx",
        "selected_Dataset_3_stat.xlsx", 
    ]
    nf_values = [7, 5, 3]

    for file_path, nf in zip(file_paths, nf_values):
        nf_output_dir = f"results_NF{nf}"
        for seed in seeds:
            np.random.seed(seed)
            data = load_data(file_path)
            if data is None:
                return
            X, y, subject_ids, feature_names = preprocess_data(data)
            search_spaces = get_search_spaces()
            models = create_models(search_spaces, RANDOM_SEED=seed)
            results = evaluate_models_loso(models, X, y, subject_ids, feature_names)
            visualize_results(results, SEED=seed, output_dir=nf_output_dir)

            print(f"\nFinal Model Comparison with N_feat: {nf} and seed:{seed}:")
            for model_name, model_results in results.items():
                if model_name == 'feature_importance':
                    continue
                print(f"\n{model_name}:")
                for metric in ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']:
                    print(f"  {metric.capitalize()}: {model_results[metric]:.4f}")
                print(f"  Best Parameters: {model_results['best_params']}")

            if 'feature_importance' in results:
                top_20 = results['feature_importance'].nlargest(20)
                print("\nXGBoost Feature Importance (Top 20):")
                for feature, importance in top_20.items():
                    print(f"  {feature}: {importance:.4f}")

if __name__ == "__main__":
    main()
#
# ^___^
# \. ./
#  \o/
#
