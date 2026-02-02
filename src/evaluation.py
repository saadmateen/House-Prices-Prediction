import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X, y):
    """Evaluate model performance with multiple metrics."""
    print("\n=== Model Evaluation ===")
    preds = model.predict(X)
    
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Additional statistics
    print(f"\nPrediction Statistics:")
    print(f"  Mean Predicted Price: ${preds.mean():,.2f}")
    print(f"  Mean Actual Price: ${y.mean():,.2f}")
    print(f"  Min Predicted: ${preds.min():,.2f}")
    print(f"  Max Predicted: ${preds.max():,.2f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}
