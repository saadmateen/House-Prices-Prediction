import joblib
import pandas as pd
import numpy as np
from src.data_preprocessing import load_data, preprocess_data

def make_predictions():
    """Generate predictions on test data using trained model."""
    print("="*60)
    print("HOUSE PRICES PREDICTION - GENERATING TEST PREDICTIONS")
    print("="*60)
    
    # Load trained model and preprocessor
    print("\n[1/4] Loading trained model and preprocessor...")
    model = joblib.load("models/house_price_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    dropped_cols = joblib.load("models/dropped_columns.pkl")
    print("Model, preprocessor, and dropped columns loaded successfully!")
    print(f"Columns to drop: {dropped_cols}")
    
    # Load test data
    print("\n[2/4] Loading test data...")
    test_df = load_data("data/test.csv")
    test_ids = test_df['Id'].copy()
    
    # Preprocess test data using same columns that were dropped in training
    print("\n[3/4] Preprocessing test data...")
    test_df = preprocess_data(test_df, drop_cols_list=dropped_cols)
    
    # Transform features using saved preprocessor
    print("Transforming features...")
    X_test = preprocessor.transform(test_df)
    print(f"Test feature matrix shape: {X_test.shape}")
    
    # Generate predictions
    print("\n[4/4] Generating predictions...")
    predictions = model.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    submission_path = "models/submission.csv"
    submission.to_csv(submission_path, index=False)
    
    print("\n=== Prediction Statistics ===")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Mean predicted price: ${predictions.mean():,.2f}")
    print(f"Median predicted price: ${np.median(predictions):,.2f}")
    print(f"Min predicted price: ${predictions.min():,.2f}")
    print(f"Max predicted price: ${predictions.max():,.2f}")
    print(f"Std deviation: ${predictions.std():,.2f}")
    
    print(f"\n=== Submission File Created ===")
    print(f"File saved to: {submission_path}")
    print(f"First 5 predictions:")
    print(submission.head())
    
    print("\n" + "="*60)
    print("PREDICTIONS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    make_predictions()
