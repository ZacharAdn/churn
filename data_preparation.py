"""
Data Preparation for Churn Prediction Model
============================================
This script prepares the data for machine learning model training:
- Handles missing values
- Encodes categorical variables
- Balances the target variable (using naive undersampling)
- Scales numerical features
- Splits data into train/test sets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DATA PREPARATION FOR CHURN PREDICTION")
print("="*80)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("\n[STEP 1] Loading data...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"✓ Loaded {len(df)} records with {df.shape[1]} features")

# =============================================================================
# STEP 2: HANDLE MISSING VALUES
# =============================================================================

print("\n[STEP 2] Handling missing values...")

# Convert TotalCharges to numeric (some values are empty strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with 0 (these are new customers with tenure=0)
missing_total_charges = df['TotalCharges'].isna().sum()
print(f"  - Found {missing_total_charges} missing values in TotalCharges")
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print(f"  ✓ Filled missing TotalCharges with 0")

# Drop customerID (not useful for prediction)
df = df.drop('customerID', axis=1)
print(f"  ✓ Dropped customerID column")

# =============================================================================
# STEP 3: ENCODE CATEGORICAL VARIABLES
# =============================================================================

print("\n[STEP 3] Encoding categorical variables...")

# Separate target variable
y = df['Churn'].map({'No': 0, 'Yes': 1})
X = df.drop('Churn', axis=1)

print(f"  Target distribution before balancing:")
print(f"    - Class 0 (No Churn): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"    - Class 1 (Churn): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")

# Binary columns (Yes/No)
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    if col in X.columns:
        X[col] = X[col].map({'No': 0, 'Yes': 1})
print(f"  ✓ Encoded {len(binary_cols)} binary columns")

# Encode gender
X['gender'] = X['gender'].map({'Male': 1, 'Female': 0})
print(f"  ✓ Encoded gender")

# Ordinal encoding for Contract (logical order: month-to-month < one year < two year)
contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
X['Contract'] = X['Contract'].map(contract_mapping)
print(f"  ✓ Encoded Contract with ordinal mapping")

# One-hot encoding for nominal categorical variables
nominal_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'PaymentMethod']

print(f"  - Applying one-hot encoding to {len(nominal_cols)} nominal columns...")
X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)
print(f"  ✓ After one-hot encoding: {X.shape[1]} features")

# =============================================================================
# STEP 4: BALANCE THE TARGET VARIABLE (NAIVE UNDERSAMPLING)
# =============================================================================

print("\n[STEP 4] Balancing target variable using undersampling...")

# Count classes
class_counts = y.value_counts()
minority_class = class_counts.min()
majority_class = class_counts.max()

print(f"  - Minority class size: {minority_class}")
print(f"  - Majority class size: {majority_class}")
print(f"  ⚠️  Will undersample majority class to match minority class")
print(f"  ⚠️  This will result in loss of {majority_class - minority_class} samples")

# Combine X and y for sampling
df_combined = pd.concat([X, y], axis=1)

# Separate by class
df_majority = df_combined[df_combined['Churn'] == 0]
df_minority = df_combined[df_combined['Churn'] == 1]

# Undersample majority class
df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)

# Combine minority class with downsampled majority class
df_balanced = pd.concat([df_majority_downsampled, df_minority])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Split back into X and y
X_balanced = df_balanced.drop('Churn', axis=1)
y_balanced = df_balanced['Churn']

print(f"  ✓ Balanced dataset size: {len(X_balanced)} samples")
print(f"  ✓ Class distribution after balancing:")
print(f"    - Class 0: {(y_balanced==0).sum()} ({(y_balanced==0).sum()/len(y_balanced)*100:.1f}%)")
print(f"    - Class 1: {(y_balanced==1).sum()} ({(y_balanced==1).sum()/len(y_balanced)*100:.1f}%)")

# =============================================================================
# STEP 5: SPLIT INTO TRAIN AND TEST SETS
# =============================================================================

print("\n[STEP 5] Splitting data into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, 
    test_size=0.2, 
    random_state=42,
    stratify=y_balanced  # Maintain class distribution
)

print(f"  ✓ Training set: {len(X_train)} samples")
print(f"  ✓ Test set: {len(X_test)} samples")
print(f"  ✓ Train/Test split: 80/20")

# =============================================================================
# STEP 6: SCALE NUMERICAL FEATURES
# =============================================================================

print("\n[STEP 6] Scaling numerical features...")

# Identify numerical columns (those that weren't one-hot encoded)
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Initialize scaler
scaler = StandardScaler()

# Fit scaler on training data and transform both train and test
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print(f"  ✓ Scaled {len(numerical_cols)} numerical features")
print(f"  ✓ Features scaled: {', '.join(numerical_cols)}")

# =============================================================================
# STEP 7: SAVE PREPARED DATA AND ARTIFACTS
# =============================================================================

print("\n[STEP 7] Saving prepared data and preprocessing artifacts...")

# Save prepared datasets
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print(f"  ✓ Saved train/test datasets")

# Save scaler for future use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ Saved StandardScaler")

# Save feature names
feature_names = X_train.columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print(f"  ✓ Saved feature names ({len(feature_names)} features)")

# =============================================================================
# STEP 8: SUMMARY AND STATISTICS
# =============================================================================

print("\n" + "="*80)
print("DATA PREPARATION SUMMARY")
print("="*80)

print(f"""
Original Dataset:
  - Total samples: {len(df)}
  - Features: {df.shape[1]}
  - Class imbalance: {class_counts[0]}/{class_counts[1]} (No/Yes)

After Balancing (Undersampling):
  - Total samples: {len(X_balanced)}
  - Samples removed: {len(df) - len(X_balanced)}
  - Class distribution: 50/50 (perfectly balanced)

Final Prepared Data:
  - Training samples: {len(X_train)}
  - Test samples: {len(X_test)}
  - Total features: {len(feature_names)}
  - Numerical features scaled: {len(numerical_cols)}
  - Categorical features encoded: {len(feature_names) - len(numerical_cols)}

Files Created:
  ✓ X_train.csv - Training features
  ✓ X_test.csv - Test features
  ✓ y_train.csv - Training labels
  ✓ y_test.csv - Test labels
  ✓ scaler.pkl - Fitted StandardScaler
  ✓ feature_names.pkl - List of feature names

Next Steps:
  1. Train machine learning models
  2. Evaluate model performance
  3. Compare different algorithms
  4. Tune hyperparameters
  5. Select best model for deployment
""")

print("="*80)
print("✓ DATA PREPARATION COMPLETE!")
print("="*80)

# Display feature importance preview
print("\n[BONUS] Feature Information:")
print(f"\nTotal Features: {len(feature_names)}")
print(f"\nFeature List:")
for i, feat in enumerate(feature_names, 1):
    print(f"  {i:2d}. {feat}")

print("\n" + "="*80)
