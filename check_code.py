from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, precision_recall_curve, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np

# APPROACH 1: Custom Scorer that optimizes threshold during CV
def recall_with_threshold_optimization(y_true, y_proba, min_precision=0.3):
    """
    Custom scorer that finds optimal threshold for maximum recall 
    while maintaining minimum precision
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find thresholds that meet minimum precision requirement
    valid_indices = precision[:-1] >= min_precision
    
    if not np.any(valid_indices):
        # If no threshold meets min precision, return recall at default threshold
        return recall[np.argmin(np.abs(thresholds - 0.5))]
    
    # Return maximum recall among valid thresholds
    valid_recalls = recall[:-1][valid_indices]
    return np.max(valid_recalls)

def custom_recall_scorer_with_threshold(estimator, X, y):
    """Custom scorer for GridSearchCV"""
    y_proba = estimator.predict_proba(X)[:, 1]
    return recall_with_threshold_optimization(y, y_proba, min_precision=0.3)

# APPROACH 2: Wrapper Class that includes threshold optimization
class ThresholdOptimizedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, min_precision=0.3):
        self.base_estimator = base_estimator
        self.min_precision = min_precision
        self.optimal_threshold_ = 0.5
        
    def fit(self, X, y):
        # Fit the base estimator
        self.base_estimator.fit(X, y)
        
        # Find optimal threshold on training data (in practice, use validation set)
        y_proba = self.base_estimator.predict_proba(X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, y_proba)
        
        # Find optimal threshold
        valid_indices = precision[:-1] >= self.min_precision
        if np.any(valid_indices):
            valid_recalls = recall[:-1][valid_indices]
            best_recall_idx = np.argmax(valid_recalls)
            # Get the index in the original arrays
            original_idx = np.where(valid_indices)[0][best_recall_idx]
            self.optimal_threshold_ = thresholds[original_idx]
        
        return self
    
    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)
    
    def predict(self, X):
        y_proba = self.predict_proba(X)[:, 1]
        return (y_proba >= self.optimal_threshold_).astype(int)

# APPROACH 3: Manual Grid Search with Threshold Optimization
def manual_hyperparameter_tuning_with_threshold():
    """
    Manual hyperparameter tuning that optimizes both model params and threshold
    """
    
    models_params = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, random_state=42), {
                'penalty': ['l1', 'l2'],
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear']
            }
        ),
        'KNN': (
            KNeighborsClassifier(), {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        )
    }
    
    from sklearn.model_selection import StratifiedKFold
    
    best_score = 0
    best_model = None
    best_params = None
    best_threshold = 0.5
    results = []
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, (base_model, param_grid) in models_params.items():
        print(f"\nTuning {model_name}...")
        
        # Generate all parameter combinations
        from sklearn.model_selection import ParameterGrid
        param_combinations = list(ParameterGrid(param_grid))
        
        for params in param_combinations:
            # Set parameters
            model = base_model.set_params(**params)
            
            # Cross-validation with threshold optimization
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Fit model
                model.fit(X_cv_train, y_cv_train)
                
                # Get probabilities
                y_proba = model.predict_proba(X_cv_val)[:, 1]
                
                # Optimize threshold and get recall
                score = recall_with_threshold_optimization(y_cv_val, y_proba, min_precision=0.3)
                cv_scores.append(score)
            
            mean_cv_score = np.mean(cv_scores)
            
            results.append({
                'Model': model_name,
                'Params': params,
                'CV_Recall_Score': mean_cv_score
            })
            
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_model = model_name
                best_params = params
    
    return results, best_model, best_params, best_score

# MAIN EXECUTION
print("="*70)
print("HYPERPARAMETER TUNING WITH THRESHOLD OPTIMIZATION")
print("="*70)

# Method 1: Using custom scorer with GridSearchCV
print("\nMethod 1: Custom Scorer with GridSearchCV")
print("-" * 40)

custom_scorer = make_scorer(custom_recall_scorer_with_threshold, 
                           greater_is_better=True, needs_proba=True)

# Example with Logistic Regression
lr_params = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear']
}

grid_lr = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    lr_params,
    cv=5,
    scoring=custom_scorer,
    n_jobs=-1
)

# Uncomment when you have X_train, y_train
# grid_lr.fit(X_train, y_train)
# print(f"Best score with threshold optimization: {grid_lr.best_score_:.4f}")
# print(f"Best parameters: {grid_lr.best_params_}")

# Method 2: Using Wrapper Class
print("\nMethod 2: Threshold-Optimized Wrapper Class")
print("-" * 40)

# Example usage of wrapper class
wrapper_lr = ThresholdOptimizedClassifier(
    LogisticRegression(max_iter=1000, random_state=42),
    min_precision=0.3
)

# This can be used in GridSearchCV like any other classifier
wrapper_params = {
    'base_estimator__penalty': ['l1', 'l2'],
    'base_estimator__C': [0.01, 0.1, 1, 10],
    'base_estimator__solver': ['liblinear'],
    'min_precision': [0.2, 0.3, 0.4]  # You can tune this too!
}

grid_wrapper = GridSearchCV(
    wrapper_lr,
    wrapper_params,
    cv=5,
    scoring='recall',  # Now we can use standard recall scoring
    n_jobs=-1
)

print("Wrapper class ready for GridSearchCV")

# Method 3: Manual approach
print("\nMethod 3: Manual Hyperparameter Tuning")
print("-" * 40)
print("Manual tuning function defined - ready to run with your data")

print("\n" + "="*70)
print("RECOMMENDATIONS FOR YOUR ATTRITION PROBLEM:")
print("="*70)
print("1. Use Method 1 (Custom Scorer) - integrates well with existing GridSearchCV")
print("2. Set min_precision based on your HR capacity (e.g., 0.3 if you can handle 3 false positives per true positive)")
print("3. This approach optimizes BOTH hyperparameters AND threshold simultaneously")
print("4. You'll get better recall than standard GridSearchCV with default threshold")