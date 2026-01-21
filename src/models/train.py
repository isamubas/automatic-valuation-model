"""
Model training script for the AVM project.
Supports multiple ML algorithms: Linear Regression, Random Forest, XGBoost, LightGBM
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Dict, Any

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from src.data.preprocess import DataPreprocessor


class AVMTrainer:
    """Trainer class for Automatic Valuation Models."""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = DataPreprocessor()
        self.results = {}
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """Train linear regression model."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        self.models['linear_regression'] = model
        self.results['linear_regression'] = metrics
        return model, metrics
    
    def train_ridge_regression(self, X_train, y_train, X_test, y_test, alpha=1.0):
        """Train Ridge regression model."""
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        self.models['ridge'] = model
        self.results['ridge'] = metrics
        return model, metrics
    
    def train_random_forest(
        self, 
        X_train, 
        y_train, 
        X_test, 
        y_test,
        n_estimators=100,
        max_depth=10,
        random_state=42
    ):
        """Train Random Forest model."""
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        self.models['random_forest'] = model
        self.results['random_forest'] = metrics
        return model, metrics
    
    def train_xgboost(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    ):
        """Train XGBoost model."""
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        self.models['xgboost'] = model
        self.results['xgboost'] = metrics
        return model, metrics
    
    def train_lightgbm(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    ):
        """Train LightGBM model."""
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        self.models['lightgbm'] = model
        self.results['lightgbm'] = metrics
        return model, metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all available models and compare performance."""
        print("Training all models...")
        
        self.train_linear_regression(X_train, y_train, X_test, y_test)
        self.train_ridge_regression(X_train, y_train, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_xgboost(X_train, y_train, X_test, y_test)
        self.train_lightgbm(X_train, y_train, X_test, y_test)
        
        print("\nModel Comparison:")
        print("-" * 60)
        for model_name, metrics in self.results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  R²: {metrics['r2']:.4f}")
        
        # Find best model
        best_model = min(self.results.items(), key=lambda x: x[1]['rmse'])
        print(f"\nBest Model: {best_model[0]} (RMSE: {best_model[1]['rmse']:.2f})")
        
        return self.models, self.results
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def save_model(self, model_name: str, filepath: Path):
        """Save trained model to file."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        print(f"Model saved to {filepath}")
    
    def save_results(self, filepath: Path):
        """Save training results to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")


if __name__ == "__main__":
    print("AVM Training utilities ready!")
    print("\nTo train models, use:")
    print("  from src.models.train import AVMTrainer")
    print("  trainer = AVMTrainer()")
    print("  trainer.train_all_models(X_train, y_train, X_test, y_test)")
