import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, RandomizedSearchCV, 
    cross_validate, train_test_split
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    make_scorer, precision_recall_curve, average_precision_score,
    fbeta_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    """
    Complete pipeline for fraud detection with model selection and hyperparameter tuning
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_params = None
        self.best_score = -np.inf
        self.results = {}
        
    def define_models(self):
        """
        Define models with their hyperparameter search spaces
        """
        models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__class_weight': ['balanced', {0: 1, 1: 10}, {0: 1, 1: 50}, {0: 1, 1: 100}],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear', 'saga']
                }
            },
            
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [10, 20, 30, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 10}],
                    'classifier__max_features': ['sqrt', 'log2', None]
                }
            },
            
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__subsample': [0.8, 0.9, 1.0]
                }
            },
            
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__min_child_weight': [1, 3, 5],
                    'classifier__gamma': [0, 0.1, 0.2],
                    'classifier__subsample': [0.8, 0.9, 1.0],
                    'classifier__colsample_bytree': [0.8, 0.9, 1.0],
                    'classifier__scale_pos_weight': [1, 10, 50, 100]  # For imbalanced data
                }
            },
            
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7, -1],
                    'classifier__num_leaves': [31, 50, 100],
                    'classifier__feature_fraction': [0.8, 0.9, 1.0],
                    'classifier__bagging_fraction': [0.8, 0.9, 1.0],
                    'classifier__class_weight': ['balanced', {0: 1, 1: 10}, {0: 1, 1: 50}]
                }
            }
        }
        
        # Add sampling strategies
        sampling_strategies = {
            'no_sampling': None,
            'smote': SMOTE(random_state=self.random_state),
            'adasyn': ADASYN(random_state=self.random_state),
            'smoteenn': SMOTEENN(random_state=self.random_state),
            'random_undersample': RandomUnderSampler(random_state=self.random_state)
        }
        
        # Combine models with sampling strategies
        self.model_configs = {}
        for model_name, model_config in models.items():
            for sampling_name, sampling_strategy in sampling_strategies.items():
                config_name = f"{model_name}_{sampling_name}"
                self.model_configs[config_name] = {
                    'model': model_config['model'],
                    'params': model_config['params'],
                    'sampling': sampling_strategy
                }
    
    def create_custom_scorers(self):
        """
        Create custom scoring functions for imbalanced fraud detection
        """
        def auc_pr_scorer(y_true, y_pred):
            return average_precision_score(y_true, y_pred)
        
        def recall_at_precision_80(y_true, y_pred):
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            idx = np.where(precision >= 0.80)[0]
            return recall[idx[-1]] if len(idx) > 0 else 0
        
        def recall_at_precision_90(y_true, y_pred):
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            idx = np.where(precision >= 0.90)[0]
            return recall[idx[-1]] if len(idx) > 0 else 0
        
        def f2_scorer(y_true, y_pred):
            # Convert probabilities to binary predictions
            y_pred_binary = (y_pred > 0.5).astype(int)
            return fbeta_score(y_true, y_pred_binary, beta=2)
        
        def business_cost_scorer(y_true, y_pred):
            y_pred_binary = (y_pred > 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            # Cost: FN = 100, FP = 1 (normalize by total samples)
            cost = (fn * 100 + fp * 1) / len(y_true)
            return -cost  # Negative because we want to minimize cost
        
        return {
            'auc_pr': make_scorer(auc_pr_scorer, needs_proba=True),
            'recall_at_80_precision': make_scorer(recall_at_precision_80, needs_proba=True),
            'recall_at_90_precision': make_scorer(recall_at_precision_90, needs_proba=True),
            'f2_score': make_scorer(f2_scorer, needs_proba=True),
            'business_cost': make_scorer(business_cost_scorer, needs_proba=True)
        }
    
    def create_pipeline(self, model, sampling_strategy=None):
        """
        Create a complete preprocessing and modeling pipeline
        """
        # Preprocessing steps
        preprocessor = ColumnTransformer([
            ('scaler', StandardScaler(), 'all')  # Scale all features
        ], remainder='passthrough')
        
        if sampling_strategy is not None:
            # Pipeline with sampling
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('sampler', sampling_strategy),
                ('classifier', model)
            ])
        else:
            # Pipeline without sampling
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
        
        return pipeline
    
    def tune_hyperparameters(self, X, y, model_name, model_config, cv_folds=5, 
                           n_iter=50, search_type='randomized'):
        """
        Tune hyperparameters for a specific model configuration
        """
        print(f"\nTuning {model_name}...")
        
        # Create pipeline
        pipeline = self.create_pipeline(
            model_config['model'], 
            model_config['sampling']
        )
        
        # Create cross-validation strategy
        cv_strategy = StratifiedKFold(
            n_splits=cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # Create custom scorers
        scorers = self.create_custom_scorers()
        
        # Choose search strategy
        if search_type == 'grid':
            search = GridSearchCV(
                pipeline,
                param_grid=model_config['params'],
                scoring=scorers['auc_pr'],  # Primary metric
                cv=cv_strategy,
                n_jobs=-1,
                verbose=1
            )
        else:  # randomized search
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=model_config['params'],
                n_iter=n_iter,
                scoring=scorers['auc_pr'],  # Primary metric
                cv=cv_strategy,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )
        
        # Fit the search
        search.fit(X, y)
        
        # Evaluate best model with all metrics
        best_pipeline = search.best_estimator_
        
        # Cross-validate with all metrics
        cv_results = cross_validate(
            best_pipeline, X, y,
            scoring=scorers,
            cv=cv_strategy,
            n_jobs=-1,
            return_train_score=False
        )
        
        # Store results
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': {
                metric: {
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                }
                for metric, scores in cv_results.items() 
                if metric.startswith('test_')
            },
            'best_estimator': best_pipeline
        }
        
        return results
    
    def compare_models(self, X, y, cv_folds=5, n_iter=30):
        """
        Compare multiple models with hyperparameter tuning
        """
        self.define_models()
        
        print("Starting comprehensive model comparison...")
        print(f"Dataset shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        print(f"Fraud rate: {np.mean(y):.4f}")
        
        all_results = {}
        
        for model_name, model_config in self.model_configs.items():
            try:
                results = self.tune_hyperparameters(
                    X, y, model_name, model_config, 
                    cv_folds=cv_folds, n_iter=n_iter
                )
                all_results[model_name] = results
                
                # Check if this is the best model so far
                current_score = results['best_score']
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_model = model_name
                    self.best_params = results['best_params']
                    
            except Exception as e:
                print(f"Error with {model_name}: {str(e)}")
                continue
        
        self.results = all_results
        return all_results
    
    def print_results_summary(self):
        """
        Print a summary of all model results
        """
        if not self.results:
            print("No results to display. Run compare_models() first.")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Create results DataFrame for easier viewing
        summary_data = []
        
        for model_name, results in self.results.items():
            row = {'Model': model_name}
            
            # Add CV metrics
            for metric_name, metric_data in results['cv_results'].items():
                clean_name = metric_name.replace('test_', '')
                row[f'{clean_name}_mean'] = f"{metric_data['mean']:.4f}"
                row[f'{clean_name}_std'] = f"{metric_data['std']:.4f}"
            
            summary_data.append(row)
        
        results_df = pd.DataFrame(summary_data)
        
        # Sort by AUC-PR (primary metric)
        if 'auc_pr_mean' in results_df.columns:
            results_df['auc_pr_mean_numeric'] = results_df['auc_pr_mean'].astype(float)
            results_df = results_df.sort_values('auc_pr_mean_numeric', ascending=False)
            results_df = results_df.drop('auc_pr_mean_numeric', axis=1)
        
        print(results_df.to_string(index=False))
        
        print(f"\n🏆 BEST MODEL: {self.best_model}")
        print(f"🎯 BEST AUC-PR SCORE: {self.best_score:.4f}")
        
        # Print best model details
        if self.best_model in self.results:
            best_results = self.results[self.best_model]
            print(f"\n📊 BEST MODEL DETAILED RESULTS:")
            print(f"Model: {self.best_model}")
            print("Cross-validation metrics:")
            for metric_name, metric_data in best_results['cv_results'].items():
                clean_name = metric_name.replace('test_', '')
                print(f"  {clean_name}: {metric_data['mean']:.4f} (±{metric_data['std']:.4f})")
            
            print(f"\nBest hyperparameters:")
            for param, value in best_results['best_params'].items():
                print(f"  {param}: {value}")
    
    def get_best_model(self):
        """
        Return the best trained model
        """
        if self.best_model and self.best_model in self.results:
            return self.results[self.best_model]['best_estimator']
        return None
    
    def evaluate_on_test_set(self, X_train, y_train, X_test, y_test):
        """
        Evaluate the best model on a held-out test set
        """
        if not self.best_model:
            print("No best model found. Run compare_models() first.")
            return None
        
        # Get best model
        best_estimator = self.get_best_model()
        
        # Train on full training set
        best_estimator.fit(X_train, y_train)
        
        # Predict on test set
        y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate all metrics
        scorers = self.create_custom_scorers()
        
        test_results = {}
        for metric_name, scorer in scorers.items():
            if 'proba' in scorer._kwargs.get('needs_proba', False):
                score = scorer._score_func(y_test, y_pred_proba)
            else:
                score = scorer._score_func(y_test, y_pred)
            test_results[metric_name] = score
        
        # Additional metrics
        test_results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        test_results['classification_report'] = classification_report(y_test, y_pred)
        
        print("\n" + "="*50)
        print("TEST SET EVALUATION")
        print("="*50)
        print(f"Model: {self.best_model}")
        print("\nTest set metrics:")
        for metric, score in test_results.items():
            if metric not in ['confusion_matrix', 'classification_report']:
                print(f"  {metric}: {score:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(test_results['confusion_matrix'])
        
        print(f"\nClassification Report:")
        print(test_results['classification_report'])
        
        return test_results

# Example usage
def main():
    """
    Example of how to use the FraudDetectionPipeline
    """
    # Simulate fraud detection dataset
    from sklearn.datasets import make_classification
    
    # Create imbalanced dataset similar to fraud detection
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.998, 0.002],  # 0.2% fraud rate
        random_state=42
    )
    
    print("Dataset created:")
    print(f"Shape: {X.shape}")
    print(f"Fraud rate: {np.mean(y):.4f}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Initialize pipeline
    fraud_pipeline = FraudDetectionPipeline(random_state=42)
    
    # Compare models (reduced iterations for demo)
    results = fraud_pipeline.compare_models(
        X_train, y_train, 
        cv_folds=3,  # Reduced for demo
        n_iter=10    # Reduced for demo
    )
    
    # Print results
    fraud_pipeline.print_results_summary()
    
    # Evaluate on test set
    test_results = fraud_pipeline.evaluate_on_test_set(
        X_train, y_train, X_test, y_test
    )
    
    return fraud_pipeline, test_results

if __name__ == "__main__":
    pipeline, test_results = main()