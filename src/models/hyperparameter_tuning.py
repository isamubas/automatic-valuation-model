"""
Hyperparameter tuning utilities for AVM models.
Uses GridSearchCV and RandomizedSearchCV for optimization.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def rmse_scorer(y_true, y_pred):
    """Custom RMSE scorer for GridSearchCV."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


class HyperparameterTuner:
    """Hyperparameter tuning class for AVM models."""
    
    def __init__(self, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42):
        """
        Initialize hyperparameter tuner.
        
        Parameters:
        -----------
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric
        n_jobs : int
            Number of parallel jobs
        random_state : int
            Random seed
        """
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params = {}
        self.best_scores = {}
    
    def tune_random_forest(
        self,
        X_train,
        y_train,
        param_grid: Optional[Dict] = None,
        method: str = 'grid',
        n_iter: int = 20
    ):
        """
        Tune Random Forest hyperparameters.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        param_grid : dict, optional
            Parameter grid. If None, uses default grid
        method : str
            'grid' for GridSearchCV, 'random' for RandomizedSearchCV
        n_iter : int
            Number of iterations for RandomizedSearchCV
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
        
        if method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        
        print(f"Tuning Random Forest ({method} search)...")
        search.fit(X_train, y_train)
        
        self.best_params['random_forest'] = search.best_params_
        self.best_scores['random_forest'] = -search.best_score_  # Convert neg MSE to RMSE
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV score (RMSE): {np.sqrt(-search.best_score_):.2f}")
        
        return search.best_estimator_, search.best_params_
    
    def tune_xgboost(
        self,
        X_train,
        y_train,
        param_grid: Optional[Dict] = None,
        method: str = 'grid',
        n_iter: int = 20
    ):
        """
        Tune XGBoost hyperparameters.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        param_grid : dict, optional
            Parameter grid. If None, uses default grid
        method : str
            'grid' for GridSearchCV, 'random' for RandomizedSearchCV
        n_iter : int
            Number of iterations for RandomizedSearchCV
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        base_model = xgb.XGBRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
        
        if method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        
        print(f"Tuning XGBoost ({method} search)...")
        search.fit(X_train, y_train)
        
        self.best_params['xgboost'] = search.best_params_
        self.best_scores['xgboost'] = -search.best_score_
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV score (RMSE): {np.sqrt(-search.best_score_):.2f}")
        
        return search.best_estimator_, search.best_params_
    
    def tune_lightgbm(
        self,
        X_train,
        y_train,
        param_grid: Optional[Dict] = None,
        method: str = 'random',
        n_iter: int = 20
    ):
        """
        Tune LightGBM hyperparameters.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        param_grid : dict, optional
            Parameter grid. If None, uses default grid
        method : str
            'grid' for GridSearchCV, 'random' for RandomizedSearchCV
        n_iter : int
            Number of iterations for RandomizedSearchCV
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 70],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        base_model = lgb.LGBMRegressor(random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1)
        
        if method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        
        print(f"Tuning LightGBM ({method} search)...")
        search.fit(X_train, y_train)
        
        self.best_params['lightgbm'] = search.best_params_
        self.best_scores['lightgbm'] = -search.best_score_
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV score (RMSE): {np.sqrt(-search.best_score_):.2f}")
        
        return search.best_estimator_, search.best_params_
    
    def tune_ridge(
        self,
        X_train,
        y_train,
        param_grid: Optional[Dict] = None,
        method: str = 'grid'
    ):
        """
        Tune Ridge regression hyperparameters.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        param_grid : dict, optional
            Parameter grid. If None, uses default grid
        method : str
            'grid' for GridSearchCV
        """
        if param_grid is None:
            param_grid = {
                'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
            }
        
        base_model = Ridge()
        
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        print("Tuning Ridge Regression...")
        search.fit(X_train, y_train)
        
        self.best_params['ridge'] = search.best_params_
        self.best_scores['ridge'] = -search.best_score_
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV score (RMSE): {np.sqrt(-search.best_score_):.2f}")
        
        return search.best_estimator_, search.best_params_


if __name__ == "__main__":
    print("Hyperparameter tuning utilities ready!")
    print("\nTo use:")
    print("  from src.models.hyperparameter_tuning import HyperparameterTuner")
    print("  tuner = HyperparameterTuner()")
    print("  best_model, best_params = tuner.tune_xgboost(X_train, y_train)")
