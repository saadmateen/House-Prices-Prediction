import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import engineer_features
from src.evaluation import evaluate_model

def train_model(X, y, model_type='random_forest'):
    """Train a regression model."""
    print(f"\n=== Training {model_type.upper()} Model ===")
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print("Training model...")
    model.fit(X, y)
    print("Training complete!")
    
    return model

def main():
    print("="*60)
    print("HOUSE PRICES PREDICTION - MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load and preprocess data
    print("\n[1/5] Loading training data...")
    df = load_data("data/train.csv")
    
    print("\n[2/5] Preprocessing data...")
    # Store columns before preprocessing to save dropped columns
    original_cols = set(df.columns)
    df = preprocess_data(df)
    remaining_cols = set(df.columns)
    
    # Calculate which columns were dropped
    dropped_cols = list(original_cols - remaining_cols - {'SalePrice'})
    print(f"\nColumns dropped during preprocessing: {dropped_cols}")
    
    print("\n[3/5] Engineering features...")
    X, y, preprocessor = engineer_features(df)
    
    # Train model
    print("\n[4/5] Training Random Forest model...")
    model = train_model(X, y, model_type='random_forest')
    
    # Evaluate model
    print("\n[5/5] Evaluating model on training data...")
    metrics = evaluate_model(model, X, y)
    
    # Save model and preprocessor
    print("\n=== Saving Model and Preprocessor ===")
    model_path = "models/house_price_model.pkl"
    preprocessor_path = "models/preprocessor.pkl"
    dropped_cols_path = "models/dropped_columns.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(dropped_cols, dropped_cols_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    print(f"Dropped columns saved to: {dropped_cols_path}")
    
    # Feature importance
    print("\n=== Top 10 Most Important Features ===")
    feature_importance = model.feature_importances_
    top_10_idx = np.argsort(feature_importance)[-10:][::-1]
    
    for i, idx in enumerate(top_10_idx, 1):
        print(f"{i}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Final Model Performance (Training Set):")
    print(f"  - RMSE: ${metrics['rmse']:,.2f}")
    print(f"  - MAE: ${metrics['mae']:,.2f}")
    print(f"  - R²: {metrics['r2']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
