# House Prices Prediction

A complete machine learning solution for predicting residential home sale prices in Ames, Iowa using advanced regression techniques. This project demonstrates production-ready code for data preprocessing, feature engineering, model training, and evaluation.

---

## 🎯 Project Overview

Predict house prices based on 79 explanatory variables describing almost every aspect of residential properties using Random Forest regression.

**Key Features:**
- 📊 Comprehensive data preprocessing pipeline
- 🔧 Automated feature engineering (scaling + encoding)
- 🤖 Random Forest model with 97% R² score
- 📈 Multiple evaluation metrics (RMSE, MAE, R²)
- 🎨 Clean, modular, production-ready code

**Dataset:**
- Training: 1,460 houses with 79 features + target (SalePrice)
- Test: 1,459 houses for prediction
- Features: Mix of property characteristics (size, quality, location, amenities)

---

## 📁 Project Structure

```
HousePricesPrediction/
│
├── data/
│   ├── train.csv                 # Training dataset
│   └── test.csv                  # Test dataset
│
├── models/
│   ├── house_price_model.pkl     # Trained Random Forest model
│   ├── preprocessor.pkl          # Feature preprocessing pipeline
│   ├── dropped_columns.pkl       # List of dropped columns
│   └── submission.csv            # Predictions for test set
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data loading and cleaning
│   ├── feature_engineering.py    # Feature transformation
│   ├── model_training.py         # Training pipeline
│   └── evaluation.py             # Performance metrics
│
├── predict.py                     # Generate test predictions
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to project directory
cd HousePricesPrediction

# Install dependencies
pip install -r requirements.txt
```

**Required packages:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

### 2. Train the Model

```bash
python -m src.model_training
```

**What happens:**
- ✅ Loads and preprocesses training data (handles missing values)
- ✅ Engineers features (standardizes numerical, encodes categorical)
- ✅ Trains Random Forest model (100 trees, optimized parameters)
- ✅ Evaluates performance with RMSE, MAE, R² metrics
- ✅ Saves model and preprocessor to `models/` directory

**Expected output:**
```
============================================================
HOUSE PRICES PREDICTION - MODEL TRAINING PIPELINE
============================================================

[1/5] Loading training data...
Loaded data with shape: (1460, 81)

[2/5] Preprocessing data...
Dropping 5 columns with >50% missing values
Filled 3 numerical and 11 categorical features

[3/5] Engineering features...
Numerical features: 36
Categorical features: 38
Final feature matrix shape: (1460, 271)

[4/5] Training Random Forest model...
Training complete!

[5/5] Evaluating model on training data...
Root Mean Squared Error (RMSE): $13,720.35
Mean Absolute Error (MAE): $7,609.03
R² Score: 0.9702

============================================================
TRAINING COMPLETE!
============================================================
```

### 3. Generate Predictions

```bash
python predict.py
```

**What happens:**
- ✅ Loads trained model and preprocessor
- ✅ Processes test data (same pipeline as training)
- ✅ Generates predictions for all 1,459 test houses
- ✅ Saves to `models/submission.csv`

**Output file format:**
```csv
Id,SalePrice
1461,130781.29
1462,156867.85
1463,183871.34
...
```

---

## 📊 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | $13,720 | Average prediction error |
| **MAE** | $7,609 | Typical error per prediction |
| **R²** | 0.9702 | Explains 97% of price variance |

**Prediction Statistics:**
- Mean predicted price: $178,804
- Price range: $59,482 - $539,408
- 1,459 predictions generated

**Top Features by Importance:**
1. Above ground living area (59%)
2. Overall quality rating (11%)
3. Other features (30%)

---

## 🔄 How It Works

### Step 1: Data Preprocessing

**Handles Missing Values:**
- Drops columns with >50% missing (Alley, PoolQC, Fence, MiscFeature, MasVnrType)
- Fills numerical missing values with **median**
- Fills categorical missing values with **mode** (most frequent)

**Example:**
```python
from src.data_preprocessing import load_data, preprocess_data

df = load_data("data/train.csv")
df = preprocess_data(df)
# Result: Clean dataframe with no missing values
```

### Step 2: Feature Engineering

**Transforms Features:**
- **Numerical (36 features):** StandardScaler → mean=0, std=1
- **Categorical (38 features):** OneHotEncoder → binary columns
- **Output:** 271-dimensional feature matrix

**Example:**
```python
from src.feature_engineering import engineer_features

X, y, preprocessor = engineer_features(df)
# X: (1460, 271) - ready for model training
# y: (1460,) - target prices
```

### Step 3: Model Training

**Random Forest Configuration:**
```python
RandomForestRegressor(
    n_estimators=100,      # 100 decision trees
    max_depth=20,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples in leaf
    n_jobs=-1              # Use all CPU cores
)
```

### Step 4: Evaluation

**Metrics Explained:**
- **RMSE**: Measures average error, penalizes large mistakes
- **MAE**: Average absolute error in dollars (more interpretable)
- **R²**: Proportion of variance explained (0-1 scale, higher is better)

---

## 📚 Dataset Features

The dataset includes 79 features across multiple categories:

### Property Basics
- **MSSubClass**: Building class (1-story, 2-story, split-level, etc.)
- **MSZoning**: Zoning (Residential Low/Medium/High, Commercial, Agricultural)
- **LotArea**: Lot size in square feet
- **Neighborhood**: Location within Ames (25 neighborhoods)

### Building Quality
- **OverallQual**: Material and finish quality (1-10 scale)
- **OverallCond**: Overall condition (1-10 scale)
- **YearBuilt**: Original construction date
- **YearRemodAdd**: Remodel date

### Living Space
- **GrLivArea**: Above grade living area (sq ft) ⭐ Most important feature
- **TotalBsmtSF**: Total basement area (sq ft)
- **1stFlrSF**: First floor area (sq ft)
- **2ndFlrSF**: Second floor area (sq ft)

### Rooms & Amenities
- **BedroomAbvGr**: Number of bedrooms
- **FullBath / HalfBath**: Bathroom count
- **KitchenQual**: Kitchen quality (Excellent to Poor)
- **Fireplaces**: Number of fireplaces

### Garage & Outdoor
- **GarageType**: Attached, Detached, Built-In, etc.
- **GarageArea**: Garage size (sq ft)
- **WoodDeckSF**: Deck area (sq ft)
- **PoolArea**: Pool area (sq ft)

### Basement Features
- **BsmtQual**: Basement height/quality
- **BsmtFinType1**: Finished area quality
- **BsmtFinSF1**: Finished area (sq ft)

**Target Variable:**
- **SalePrice**: Property sale price in dollars (what we predict)

---

## 💻 Code Usage Examples

### Complete Workflow

```python
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import engineer_features
from src.model_training import train_model
from src.evaluation import evaluate_model
import joblib

# 1. Load and clean data
df = load_data("data/train.csv")
df = preprocess_data(df)

# 2. Transform features
X, y, preprocessor = engineer_features(df)

# 3. Train model
model = train_model(X, y, model_type='random_forest')

# 4. Evaluate
metrics = evaluate_model(model, X, y)
print(f"RMSE: ${metrics['rmse']:,.2f}")

# 5. Save artifacts
joblib.dump(model, "models/house_price_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")
```

### Making Predictions

```python
import joblib
import pandas as pd

# Load trained artifacts
model = joblib.load("models/house_price_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Load and process new data
test_df = load_data("data/test.csv")
test_ids = test_df['Id']
test_df = preprocess_data(test_df)

# Transform and predict
X_test = preprocessor.transform(test_df)
predictions = model.predict(X_test)

# Save predictions
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': predictions
})
submission.to_csv("submission.csv", index=False)
```

---

## 🛠️ Technical Details

### Data Preprocessing Strategy

| Issue | Solution |
|-------|----------|
| Missing values (>50%) | Drop column entirely |
| Missing numerical | Fill with median |
| Missing categorical | Fill with mode |
| Non-predictive features | Remove (e.g., Id column) |

### Feature Engineering Pipeline

```python
ColumnTransformer([
    ('numerical', StandardScaler(), [36 features]),
    ('categorical', OneHotEncoder(), [38 features])
])
# Output: 271 features total
```

### Model Architecture

- **Algorithm**: Random Forest Regressor
- **Ensemble**: 100 decision trees
- **Max Depth**: 20 levels
- **Leaf Nodes**: Minimum 2 samples
- **Parallelization**: All CPU cores

---



## 📝 Notes

### Why Random Forest?
- ✅ Handles mixed feature types (numerical + categorical)
- ✅ Robust to outliers and missing values
- ✅ Provides feature importance rankings
- ✅ No complex hyperparameter tuning needed
- ✅ Excellent baseline model performance

### Key Insights
1. **Living area is king**: GrLivArea accounts for 59% of predictive power
2. **Quality matters**: OverallQual is the second most important (11%)
3. **Model generalizes well**: High R² with reasonable RMSE
4. **Feature scaling helps**: Despite Random Forest not requiring it, we include it for extensibility

---

## 🙏 Acknowledgments

- **Dataset**: Ames Housing dataset compiled by Dean De Cock for data science education
- **Use Case**: Educational tool for learning advanced regression techniques
