import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def engineer_features(df: pd.DataFrame):
    """Apply feature engineering and encoding."""
    print("\n=== Feature Engineering ===")
    
    # Separate features and target if target present
    target = None
    if 'SalePrice' in df.columns:
        target = df['SalePrice']
        df = df.drop(['SalePrice'], axis=1)
        print(f"Target variable extracted. Mean SalePrice: ${target.mean():,.2f}")
    
    # Identify categorical and numerical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Numerical features: {len(num_cols)}")
    print(f"Categorical features: {len(cat_cols)}")
    
    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # Fit and transform features
    print("Fitting and transforming features...")
    features = preprocessor.fit_transform(df)
    print(f"Final feature matrix shape: {features.shape}")
    
    return features, target, preprocessor
