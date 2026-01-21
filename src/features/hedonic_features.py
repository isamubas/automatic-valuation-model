"""
Hedonic feature engineering for real estate valuation models.
Creates property characteristics-based features for hedonic pricing models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class HedonicFeatureEngineer:
    """
    Feature engineering class for hedonic real estate models.
    Creates features based on property characteristics.
    """
    
    def __init__(self):
        self.feature_names = []
    
    def create_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from structural characteristics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with columns: bed, bath, house_size, acre_lot
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with additional structural features
        """
        df = df.copy()
        
        # Price per square foot (if price available)
        if 'price' in df.columns and 'house_size' in df.columns:
            df['price_per_sqft'] = df['price'] / (df['house_size'] + 1)  # +1 to avoid division by zero
        
        # Bedroom to bathroom ratio
        if 'bed' in df.columns and 'bath' in df.columns:
            df['bed_bath_ratio'] = df['bed'] / (df['bath'] + 0.5)  # +0.5 to avoid division by zero
        
        # Total rooms
        if 'bed' in df.columns and 'bath' in df.columns:
            df['total_rooms'] = df['bed'] + df['bath']
        
        # Size per bedroom
        if 'house_size' in df.columns and 'bed' in df.columns:
            df['sqft_per_bedroom'] = df['house_size'] / (df['bed'] + 1)
        
        # Size per bathroom
        if 'house_size' in df.columns and 'bath' in df.columns:
            df['sqft_per_bathroom'] = df['house_size'] / (df['bath'] + 1)
        
        # Lot size per house size (density indicator)
        if 'acre_lot' in df.columns and 'house_size' in df.columns:
            # Convert acres to square feet (1 acre = 43,560 sqft)
            df['lot_size_sqft'] = df['acre_lot'] * 43560
            df['lot_to_house_ratio'] = df['lot_size_sqft'] / (df['house_size'] + 1)
        
        return df
    
    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from location data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with columns: city, state, zip_code
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with additional location features
        """
        df = df.copy()
        
        # Create location identifier
        if 'city' in df.columns and 'state' in df.columns:
            df['city_state'] = df['city'].astype(str) + ', ' + df['state'].astype(str)
        
        # Extract first digit of zip code (region indicator)
        if 'zip_code' in df.columns:
            # Convert to numeric first, handling NaN values
            zip_numeric = pd.to_numeric(df['zip_code'], errors='coerce')
            # Fill NaN with 0, then convert to string for extraction
            zip_str = zip_numeric.fillna(0).astype(int).astype(str).str.zfill(5)
            # Extract first digit
            df['zip_first_digit'] = pd.to_numeric(zip_str.str[0], errors='coerce').fillna(0)
            # Extract first 3 digits for region
            df['zip_region'] = pd.to_numeric(zip_str.str[:3], errors='coerce').fillna(0)
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from temporal data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with prev_sold_date column
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with additional temporal features
        """
        df = df.copy()
        
        if 'prev_sold_date' in df.columns:
            # Convert to datetime if not already
            try:
                df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
                
                # Days since last sale
                df['days_since_sale'] = (pd.Timestamp.now() - df['prev_sold_date']).dt.days
                df['days_since_sale'] = df['days_since_sale'].fillna(-1)  # -1 for never sold
                
                # Year of last sale
                df['year_sold'] = df['prev_sold_date'].dt.year
                df['year_sold'] = df['year_sold'].fillna(0)  # 0 for never sold
                
                # Month of last sale
                df['month_sold'] = df['prev_sold_date'].dt.month
                df['month_sold'] = df['month_sold'].fillna(0)  # 0 for never sold
                
                # Has been sold before (binary)
                df['has_sale_history'] = df['prev_sold_date'].notna().astype(int)
                
            except Exception as e:
                print(f"Warning: Could not process prev_sold_date: {e}")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different property characteristics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with interaction features
        """
        df = df.copy()
        
        # Size * bedrooms (total living space indicator)
        if 'house_size' in df.columns and 'bed' in df.columns:
            df['size_bed_interaction'] = df['house_size'] * df['bed']
        
        # Size * bathrooms
        if 'house_size' in df.columns and 'bath' in df.columns:
            df['size_bath_interaction'] = df['house_size'] * df['bath']
        
        # Lot size * bedrooms (spaciousness indicator)
        if 'acre_lot' in df.columns and 'bed' in df.columns:
            df['lot_bed_interaction'] = df['acre_lot'] * df['bed']
        
        return df
    
    def engineer_all_features(
        self, 
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        exclude_columns : list, optional
            Columns to exclude from feature engineering
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with all engineered features
        """
        df = df.copy()
        
        if exclude_columns is None:
            exclude_columns = []
        
        # Store original columns to exclude from feature engineering
        original_cols = set(df.columns)
        
        # Apply all feature engineering steps
        df = self.create_structural_features(df)
        df = self.create_location_features(df)
        df = self.create_temporal_features(df)
        df = self.create_interaction_features(df)
        
        # Remove price_per_sqft if price is in exclude_columns (to avoid data leakage)
        if 'price' in exclude_columns and 'price_per_sqft' in df.columns:
            df = df.drop(columns=['price_per_sqft'])
        
        # Store feature names
        new_features = [col for col in df.columns if col not in original_cols]
        self.feature_names = new_features
        
        print(f"Created {len(new_features)} new features:")
        for feat in new_features[:10]:  # Show first 10
            print(f"  - {feat}")
        if len(new_features) > 10:
            print(f"  ... and {len(new_features) - 10} more")
        
        return df
    
    def get_feature_importance_categories(self) -> dict:
        """
        Get categories of features for interpretation.
        
        Returns:
        --------
        dict
            Dictionary mapping feature categories to feature names
        """
        categories = {
            'structural': ['bed', 'bath', 'house_size', 'acre_lot', 'bed_bath_ratio', 
                          'total_rooms', 'sqft_per_bedroom', 'sqft_per_bathroom', 
                          'lot_to_house_ratio'],
            'location': ['city', 'state', 'zip_code', 'city_state', 'zip_first_digit', 'zip_region'],
            'temporal': ['prev_sold_date', 'days_since_sale', 'year_sold', 'month_sold', 
                        'has_sale_history'],
            'interaction': ['size_bed_interaction', 'size_bath_interaction', 'lot_bed_interaction'],
            'other': ['brokered_by', 'status']
        }
        return categories


if __name__ == "__main__":
    print("Hedonic feature engineering utilities ready!")
    print("\nTo use:")
    print("  from src.features.hedonic_features import HedonicFeatureEngineer")
    print("  engineer = HedonicFeatureEngineer()")
    print("  df_engineered = engineer.engineer_all_features(df)")
