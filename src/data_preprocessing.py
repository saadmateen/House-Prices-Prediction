import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded data with shape: {df.shape}")
    return df

def preprocess_data(df: pd.DataFrame, drop_cols_list=None) -> pd.DataFrame:
    """Basic cleaning and handling missing values.
    
    Args:
        df: Input dataframe
        drop_cols_list: Optional list of columns to drop (for consistency with training)
    """
    print(f"Starting preprocessing with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Drop Id column if present (not useful for prediction)
    if 'Id' in df.columns:
        df = df.drop(['Id'], axis=1)
    
    # If specific columns to drop are provided, use those
    if drop_cols_list is not None:
        print(f"Dropping specified columns: {drop_cols_list}")
        df = df.drop(drop_cols_list, axis=1, errors='ignore')
    else:
        # Drop columns with too many missing values (>50%) - for training only
        missing_threshold = 0.5
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
        print(f"Dropping {len(cols_to_drop)} columns with >50% missing values: {cols_to_drop}")
        df = df.drop(cols_to_drop, axis=1, errors='ignore')
    
    # Fill missing numerical values with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df.loc[:, col] = df[col].fillna(median_val)
            print(f"Filled {col} missing values with median: {median_val}")
    
    # Fill missing categorical values with mode or 'None'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            if len(df[col].mode()) > 0:
                mode_val = df[col].mode()[0]
                df.loc[:, col] = df[col].fillna(mode_val)
                print(f"Filled {col} missing values with mode: {mode_val}")
            else:
                df.loc[:, col] = df[col].fillna('None')
                print(f"Filled {col} missing values with 'None'")
    
    print(f"Preprocessing complete. Final shape: {df.shape}")
    return df
