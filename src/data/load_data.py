"""
Data loading utilities for the AVM project.
Supports multiple file formats: CSV, Excel, Parquet, JSON
"""

import pandas as pd
import os
from pathlib import Path
from typing import Union, Optional


def load_dataset(
    file_path: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load dataset from various file formats.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the dataset file
    file_type : str, optional
        File type ('csv', 'excel', 'parquet', 'json'). 
        If None, inferred from file extension
    **kwargs : dict
        Additional arguments passed to pandas read functions
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Infer file type from extension if not provided
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')
    
    # Load based on file type
    if file_type in ['csv']:
        df = pd.read_csv(file_path, **kwargs)
    elif file_type in ['xlsx', 'xls', 'excel']:
        df = pd.read_excel(file_path, **kwargs)
    elif file_type in ['parquet']:
        df = pd.read_parquet(file_path, **kwargs)
    elif file_type in ['json']:
        df = pd.read_json(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_data_path(file_name: str, subfolder: str = "raw") -> Path:
    """
    Get the full path to a data file.
    
    Parameters:
    -----------
    file_name : str
        Name of the data file
    subfolder : str
        Subfolder within data directory ('raw' or 'processed')
    
    Returns:
    --------
    Path
        Full path to the data file
    """
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / subfolder / file_name
    return data_path


if __name__ == "__main__":
    # Example usage
    print("Data loading utilities ready!")
    print("\nTo load your dataset, use:")
    print("  from src.data.load_data import load_dataset")
    print("  df = load_dataset('your_file.csv')")
