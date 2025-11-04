# Car Price Prediction Machine Learning Project
# Complete Training Script with Data Preprocessing, Model Building & Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CAR PRICE PREDICTION - MACHINE LEARNING PROJECT")
print("="*80)

# ============================================================================
# SECTION 1: GENERATE SAMPLE DATASET
# ============================================================================
# Creating a realistic car dataset with 1000 samples
# Features: Brand, Model_Year, Fuel_Type, Transmission, Mileage, Engine_Size

print("\n[STEP 1] Generating Sample Car Dataset...")

np.random.seed(42)
n_samples = 1000

# Define categorical features
brands = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Kia', 'Nissan', 'Chevrolet']
fuel_types = ['Petrol', 'Diesel', 'Electric', 'Hybrid']
transmissions = ['Manual', 'Automatic']

# Generate data
data = {
    'Brand': np.random.choice(brands, n_samples),
    'Model_Year': np.random.randint(2010, 2024, n_samples),
    'Fuel_Type': np.random.choice(fuel_types, n_samples, p=[0.4, 0.3, 0.15, 0.15]),
    'Transmission': np.random.choice(transmissions, n_samples, p=[0.4, 0.6]),
    'Mileage': np.random.randint(5000, 150000, n_samples),
    'Engine_Size': np.round(np.random.uniform(1.0, 5.0, n_samples), 1)
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate realistic prices based on features
# Price formula considers: brand value, year, mileage, engine size, fuel type
brand_multiplier = {
    'Toyota': 1.0, 'Honda': 1.0, 'Ford': 0.9, 'BMW': 1.5, 
    'Mercedes': 1.6, 'Audi': 1.4, 'Hyundai': 0.85, 
    'Kia': 0.8, 'Nissan': 0.9, 'Chevrolet': 0.85
}

fuel_multiplier = {'Petrol': 1.0, 'Diesel': 1.1, 'Electric': 1.3, 'Hybrid': 1.2}
transmission_multiplier = {'Manual': 0.95, 'Automatic': 1.05}

# Calculate base price
df['Price'] = 0
for idx, row in df.iterrows():
    base_price = 15000
    base_price *= brand_multiplier[row['Brand']]
    base_price *= fuel_multiplier[row['Fuel_Type']]
    base_price *= transmission_multiplier[row['Transmission']]
    base_price += (row['Model_Year'] - 2010) * 1000  # Newer cars cost more
    base_price -= (row['Mileage'] / 1000) * 50  # Higher mileage reduces price
    base_price += row['Engine_Size'] * 2000  # Larger engines cost more
    
    # Add some random noise
    base_price *= np.random.uniform(0.9, 1.1)
    df.loc[idx, 'Price'] = max(5000, base_price)  # Minimum price $5000

df['Price'] = df['Price'].round(0).astype(int)

# Introduce 2% missing values randomly
mask = np.random.random(df.shape) < 0.02
df_with_missing = df.mask(mask)

# Save dataset
df_with_missing.to_csv('car_dataset.csv', index=False)
print(f"✓ Dataset created with {n_samples} samples")
print(f"✓ Saved as 'car_dataset.csv'")

# ============================================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================================
print("\n[STEP 2] Data Preprocessing...")

# Load dataset
df = pd.read_csv('car_dataset.csv')

print(f"\nDataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\n[STEP 2.1] Checking Missing Values...")
missing_values = df.isnull().sum()
print("\nMissing Values Count:")
print(missing_values[missing_values > 0])

# Handle missing values
print("\n[STEP 2.2] Handling Missing Values...")

# For numerical columns: fill with median
numerical_cols = ['Model_Year', 'Mileage', 'Engine_Size', 'Price']
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  ✓ Filled {col} missing values with median")

# For categorical columns: fill with mode
categorical_cols = ['Brand', 'Fuel_Type', 'Transmission']
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"  ✓ Filled {col} missing values with mode")

print(f"\n✓ Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")

# ============================================================================
# SECTION 3: FEATURE ENCODING
# ============================================================================
print("\n[STEP 3] Encoding Categorical Features...")

# Create a copy for encoding
df_encoded = df.copy()

# Label Encoding for categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
    print(f"  ✓ Encoded {col}: {list(le.classes_)}")

# Save label encoders for later use in Flask app
joblib.dump(label_encoders, 'label_encoders.pkl')
print("\n✓ Label encoders saved as 'label_encoders.pkl'")

# ============================================================================
# SECTION 4: FEATURE SCALING & TRAIN-TEST SPLIT
# ============================================================================
print("\n[STEP 4] Preparing Data for Training...")

# Separate features and target
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for later use
joblib.dump(scaler, 'scaler.pkl')
print("✓ Feature scaler saved as 'scaler.pkl'")

# ============================================================================
# SECTION 5: MODEL BUILDING & TRAINING
# ============================================================================
print("\n[STEP 5] Building and Training ML Models...")
print("-" * 80)

# Dictionary to store models and results
models = {}
results = []

# Model 1: Linear Regression
print("\n[Model 1] Linear Regression")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

models['Linear Regression'] = lr_model
results.append({
    'Model': 'Linear Regression',
    'R² Score': lr_r2,
    'MAE': lr_mae,
    'RMSE': lr_rmse
})

print(f"  R² Score: {lr_r2:.4f}")
print(f"  MAE: ${lr_mae:.2f}")
print(f"  RMSE: ${lr_rmse:.2f}")

# Model 2: Decision Tree Regressor
print("\n[Model 2] Decision Tree Regressor")
dt_model = DecisionTreeRegressor(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

dt_r2 = r2_score(y_test, dt_pred)
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))

models['Decision Tree'] = dt_model
results.append({
    'Model': 'Decision Tree',
    'R² Score': dt_r2,
    'MAE': dt_mae,
    'RMSE': dt_rmse
})

print(f"  R² Score: {dt_r2:.4f}")
print(f"  MAE: ${dt_mae:.2f}")
print(f"  RMSE: ${dt_rmse:.2f}")

# Model 3: Random Forest Regressor
print("\n[Model 3] Random Forest Regressor")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

models['Random Forest'] = rf_model
results.append({
    'Model': 'Random Forest',
    'R² Score': rf_r2,
    'MAE': rf_mae,
    'RMSE': rf_rmse
})

print(f"  R² Score: {rf_r2:.4f}")
print(f"  MAE: ${rf_mae:.2f}")
print(f"  RMSE: ${rf_rmse:.2f}")

# Model 4: XGBoost Regressor
print("\n[Model 4] XGBoost Regressor")
xgb_model = XGBRegressor(n_estimators=100, random_state=42, max_depth=10, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

xgb_r2 = r2_score(y_test, xgb_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

models['XGBoost'] = xgb_model
results.append({
    'Model': 'XGBoost',
    'R² Score': xgb_r2,
    'MAE': xgb_mae,
    'RMSE': xgb_rmse
})

print(f"  R² Score: {xgb_r2:.4f}")
print(f"  MAE: ${xgb_mae:.2f}")
print(f"  RMSE: ${xgb_rmse:.2f}")

# ============================================================================
# SECTION 6: MODEL COMPARISON
# ============================================================================
print("\n[STEP 6] Model Performance Comparison")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R² Score', ascending=False)
print("\n", results_df.to_string(index=False))

# Identify best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² Score: {results_df.iloc[0]['R² Score']:.4f}")
print(f"   MAE: ${results_df.iloc[0]['MAE']:.2f}")
print(f"   RMSE: ${results_df.iloc[0]['RMSE']:.2f}")

# ============================================================================
# SECTION 7: VISUALIZATIONS
# ============================================================================
print("\n[STEP 7] Creating Visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# Visualization 1: Correlation Heatmap
plt.subplot(2, 3, 1)
correlation_matrix = df_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()

# Visualization 2: Model Performance Comparison (R² Score)
plt.subplot(2, 3, 2)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
plt.barh(results_df['Model'], results_df['R² Score'], color=colors)
plt.xlabel('R² Score', fontweight='bold')
plt.title('Model Performance Comparison (R² Score)', fontsize=14, fontweight='bold')
plt.xlim(0, 1)
for i, v in enumerate(results_df['R² Score']):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center')

# Visualization 3: MAE Comparison
plt.subplot(2, 3, 3)
plt.barh(results_df['Model'], results_df['MAE'], color=colors)
plt.xlabel('Mean Absolute Error ($)', fontweight='bold')
plt.title('Model Performance Comparison (MAE)', fontsize=14, fontweight='bold')
for i, v in enumerate(results_df['MAE']):
    plt.text(v + 50, i, f'${v:.0f}', va='center')

# Visualization 4: Feature Importance (Random Forest)
plt.subplot(2, 3, 4)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='#95E1D3')
plt.xlabel('Importance', fontweight='bold')
plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
for i, v in enumerate(feature_importance['Importance']):
    plt.text(v + 0.005, i, f'{v:.3f}', va='center')

# Visualization 5: Predicted vs Actual (Best Model)
plt.subplot(2, 3, 5)
if best_model_name == 'Linear Regression':
    best_pred = lr_pred
elif best_model_name == 'Decision Tree':
    best_pred = dt_pred
elif best_model_name == 'Random Forest':
    best_pred = rf_pred
else:
    best_pred = xgb_pred

plt.scatter(y_test, best_pred, alpha=0.5, color='#F38181')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)', fontweight='bold')
plt.ylabel('Predicted Price ($)', fontweight='bold')
plt.title(f'Predicted vs Actual Prices ({best_model_name})', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

# Visualization 6: Residual Distribution
plt.subplot(2, 3, 6)
residuals = y_test - best_pred
plt.hist(residuals, bins=30, color='#AA96DA', edgecolor='black', alpha=0.7)
plt.xlabel('Residual (Actual - Predicted)', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title(f'Residual Distribution ({best_model_name})', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation_visualizations.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'model_evaluation_visualizations.png'")

plt.show()

# ============================================================================
# SECTION 8: SAVE THE BEST MODEL
# ============================================================================
print("\n[STEP 8] Saving the Best Model...")

# Save the best model
joblib.dump(best_model, 'car_price_model.pkl')
print(f"✓ Best model ({best_model_name}) saved as 'car_price_model.pkl'")

# Save model metadata
model_metadata = {
    'model_name': best_model_name,
    'r2_score': results_df.iloc[0]['R² Score'],
    'mae': results_df.iloc[0]['MAE'],
    'rmse': results_df.iloc[0]['RMSE'],
    'features': list(X.columns),
    'trained_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}
joblib.dump(model_metadata, 'model_metadata.pkl')
print("✓ Model metadata saved as 'model_metadata.pkl'")

# ============================================================================
# SECTION 9: PROJECT SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PROJECT SUMMARY")
print("="*80)

print("\n1. DATASET:")
print(f"   - Total Samples: {len(df)}")
print(f"   - Features: {', '.join(X.columns)}")
print(f"   - Target Variable: Price")
print(f"   - Price Range: ${df['Price'].min():,.0f} - ${df['Price'].max():,.0f}")

print("\n2. PREPROCESSING STEPS:")
print("   ✓ Handled missing values (filled with median/mode)")
print("   ✓ Encoded categorical features (Brand, Fuel_Type, Transmission)")
print("   ✓ Applied feature scaling (StandardScaler)")
print("   ✓ Split data: 80% training, 20% testing")

print("\n3. MODELS TRAINED:")
for idx, row in results_df.iterrows():
    print(f"   {row['Model']:20} - R²: {row['R² Score']:.4f}, MAE: ${row['MAE']:.2f}")

print(f"\n4. BEST MODEL: {best_model_name}")
print(f"   - Explanation: {best_model_name} achieved the highest R² score of {results_df.iloc[0]['R² Score']:.4f}")
print(f"   - This means the model explains {results_df.iloc[0]['R² Score']*100:.2f}% of the variance in car prices")
print(f"   - Average prediction error (MAE): ${results_df.iloc[0]['MAE']:.2f}")

print("\n5. KEY OBSERVATIONS:")
print(f"   - Most Important Feature: {feature_importance.iloc[0]['Feature']}")
print(f"   - Feature Importance: {feature_importance.iloc[0]['Importance']:.3f}")
print("   - Model Performance: Excellent predictions with low error rates")
print("   - The model successfully captures pricing patterns based on car features")

print("\n6. FILES GENERATED:")
print("   ✓ car_dataset.csv - Original dataset")
print("   ✓ car_price_model.pkl - Trained best model")
print("   ✓ label_encoders.pkl - Categorical encoders")
print("   ✓ scaler.pkl - Feature scaler")
print("   ✓ model_metadata.pkl - Model information")
print("   ✓ model_evaluation_visualizations.png - Performance charts")

print("\n7. NEXT STEPS:")
print("   → Deploy the model using Flask web app (app.py)")
print("   → Users can input car features and get price predictions")
print("   → Model ready for production use!")

print("\n" + "="*80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)