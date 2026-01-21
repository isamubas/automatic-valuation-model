"""
Data preprocessing utilities for the AVM project.
Handles missing values, outliers, feature engineering, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Preprocessing pipeline for AVM datasets."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = None
        self.target_column = None
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str = "median",
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        strategy : str
            Strategy for imputation ('mean', 'median', 'mode', 'drop')
        columns : list, optional
            Specific columns to process. If None, processes all columns
        """
        df = df.copy()
        
        if columns is None:
            columns = df.columns.tolist()
        
        for col in columns:
            if df[col].isnull().sum() > 0:
                if strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif strategy == "drop":
                    df.dropna(subset=[col], inplace=True)
        
        return df
    
    def remove_outliers(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        factor: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers using IQR or Z-score method.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list, optional
            Columns to check for outliers
        method : str
            Method to use ('iqr' or 'zscore')
        factor : float
            Factor for outlier detection (IQR multiplier or Z-score threshold)
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == "iqr":
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == "zscore":
            for col in columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < factor]
        
        return df
    
    def encode_categorical(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = "label"
    ) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list, optional
            Categorical columns to encode
        method : str
            Encoding method ('label' or 'onehot')
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if method == "label":
            for col in columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.encoders[col].transform(df[col].astype(str))
        
        elif method == "onehot":
            df = pd.get_dummies(df, columns=columns, prefix=columns)
        
        return df
    
    def scale_features(
        self, 
        X: pd.DataFrame, 
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        fit : bool
            Whether to fit the scaler (True for training, False for prediction)
        """
        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            for col in numeric_cols:
                self.scalers[col] = StandardScaler()
                X[col] = self.scalers[col].fit_transform(X[[col]])
        else:
            for col in numeric_cols:
                if col in self.scalers:
                    X[col] = self.scalers[col].transform(X[[col]])
        
        return X
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Complete data preparation pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_column : str
            Name of the target variable column
        feature_columns : list, optional
            List of feature columns. If None, uses all except target
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility
        """
        df = df.copy()
        
        # Store column information
        self.target_column = target_column
        if feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != target_column]
        else:
            self.feature_columns = feature_columns
        
        # Separate features and target
        X = df[self.feature_columns]
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_columns
        }


if __name__ == "__main__":
    print("Data preprocessing utilities ready!")
    print("\nExample usage:")
    print("  from src.data.preprocess import DataPreprocessor")
    print("  preprocessor = DataPreprocessor()")
    print("  df_clean = preprocessor.handle_missing_values(df)")
