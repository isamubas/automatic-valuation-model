"""
Exploratory Data Analysis (EDA) utilities for real estate datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


class RealEstateEDA:
    """EDA class for real estate datasets."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA with dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to analyze
        """
        self.df = df.copy()
    
    def summary_statistics(self, target_column: str = 'price'):
        """
        Display summary statistics for the dataset.
        
        Parameters:
        -----------
        target_column : str
            Name of target variable
        """
        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Total Records: {len(self.df):,}")
        print(f"Total Features: {len(self.df.columns)}")
        
        print(f"\nMissing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
        else:
            print("  No missing values!")
        
        print(f"\nData Types:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count}")
        
        if target_column in self.df.columns:
            print(f"\nTarget Variable ({target_column}) Statistics:")
            print(self.df[target_column].describe())
    
    def plot_target_distribution(self, target_column: str = 'price', figsize=(12, 5)):
        """
        Plot distribution of target variable.
        
        Parameters:
        -----------
        target_column : str
            Name of target variable
        figsize : tuple
            Figure size
        """
        if target_column not in self.df.columns:
            print(f"Error: {target_column} not found in dataset")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(self.df[target_column].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel(target_column.title())
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of {target_column.title()}')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(self.df[target_column].dropna(), vert=True)
        axes[1].set_ylabel(target_column.title())
        axes[1].set_title(f'Box Plot of {target_column.title()}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_distributions(self, columns: Optional[List[str]] = None, figsize=(15, 10)):
        """
        Plot distributions of key features.
        
        Parameters:
        -----------
        columns : list, optional
            Columns to plot. If None, plots numerical columns
        figsize : tuple
            Figure size
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to top 9 features for readability
        columns = columns[:9]
        
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for idx, col in enumerate(columns):
            if col in self.df.columns:
                data = self.df[col].dropna()
                axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'{col}')
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def correlation_heatmap(self, target_column: str = 'price', figsize=(12, 10)):
        """
        Plot correlation heatmap.
        
        Parameters:
        -----------
        target_column : str
            Name of target variable
        figsize : tuple
            Figure size
        """
        # Select only numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            print("Not enough numerical columns for correlation analysis")
            return
        
        corr_matrix = self.df[numerical_cols].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
        # Show top correlations with target
        if target_column in numerical_cols:
            print(f"\nTop Correlations with {target_column}:")
            correlations = corr_matrix[target_column].sort_values(ascending=False)
            correlations = correlations[correlations.index != target_column]
            print(correlations.head(10).to_string())
    
    def plot_target_vs_features(
        self,
        target_column: str = 'price',
        feature_columns: Optional[List[str]] = None,
        figsize=(15, 10)
    ):
        """
        Plot target variable against key features.
        
        Parameters:
        -----------
        target_column : str
            Name of target variable
        feature_columns : list, optional
            Features to plot against target
        figsize : tuple
            Figure size
        """
        if target_column not in self.df.columns:
            print(f"Error: {target_column} not found in dataset")
            return
        
        if feature_columns is None:
            # Select top numerical features (excluding target)
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numerical_cols if col != target_column][:6]
        
        n_cols = 3
        n_rows = (len(feature_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for idx, col in enumerate(feature_columns):
            if col in self.df.columns:
                axes[idx].scatter(
                    self.df[col],
                    self.df[target_column],
                    alpha=0.3,
                    s=10
                )
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel(target_column)
                axes[idx].set_title(f'{target_column} vs {col}')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(feature_columns), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def categorical_analysis(self, columns: Optional[List[str]] = None, target_column: str = 'price'):
        """
        Analyze categorical variables.
        
        Parameters:
        -----------
        columns : list, optional
            Categorical columns to analyze
        target_column : str
            Target variable for comparison
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in columns:
            if col in self.df.columns:
                print(f"\n{col.upper()}:")
                print("-" * 40)
                value_counts = self.df[col].value_counts()
                print(f"Unique values: {self.df[col].nunique()}")
                print(f"\nTop 10 values:")
                print(value_counts.head(10).to_string())
                
                if target_column in self.df.columns:
                    # Average target by category
                    if self.df[col].nunique() <= 20:  # Only if reasonable number of categories
                        avg_by_category = self.df.groupby(col)[target_column].mean().sort_values(ascending=False)
                        print(f"\nAverage {target_column} by {col}:")
                        print(avg_by_category.to_string())


if __name__ == "__main__":
    print("EDA utilities ready!")
    print("\nTo use:")
    print("  from src.eda.exploratory_analysis import RealEstateEDA")
    print("  eda = RealEstateEDA(df)")
    print("  eda.summary_statistics()")
