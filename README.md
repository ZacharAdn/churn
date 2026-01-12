# Telco Customer Churn Prediction Project

## Project Overview
This project aims to predict customer churn (customer attrition) for a telecommunications company using machine learning techniques.

## Dataset Information
- **Source**: Kaggle - Telco Customer Churn Dataset
- **Size**: 7,043 customers with 21 features
- **Target Variable**: Churn (Yes/No)
- **Class Distribution**: 
  - No Churn: 73.46%
  - Churn: 26.54%

## Features

### 1. Customer Demographics (5 features)
- `customerID`: Unique identifier
- `gender`: Male/Female
- `SeniorCitizen`: Whether customer is 65+ (0/1)
- `Partner`: Has a partner (Yes/No)
- `Dependents`: Has dependents (Yes/No)

### 2. Service Features (11 features)
- `tenure`: Months with the company (0-72)
- `PhoneService`: Has phone service (Yes/No)
- `MultipleLines`: Has multiple phone lines
- `InternetService`: Type (DSL/Fiber optic/No)
- `OnlineSecurity`: Has online security add-on
- `OnlineBackup`: Has online backup add-on
- `DeviceProtection`: Has device protection add-on
- `TechSupport`: Has tech support add-on
- `StreamingTV`: Has streaming TV add-on
- `StreamingMovies`: Has streaming movies add-on

### 3. Billing Features (5 features)
- `Contract`: Type (Month-to-month/One year/Two year)
- `PaperlessBilling`: Uses paperless billing (Yes/No)
- `PaymentMethod`: Payment method used
- `MonthlyCharges`: Monthly charge amount ($)
- `TotalCharges`: Total amount charged ($)

## Setup Instructions

### 1. Create Virtual Environment
```bash
python3 -m venv churn_venv
source churn_venv/bin/activate  # On Windows: churn_venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
The dataset has already been downloaded. If you need to re-download:
```bash
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip
```

## Scripts

### data_exploration.py
Comprehensive exploratory data analysis script that:
- Displays dataset structure and statistics
- Analyzes all features by category
- Identifies data quality issues
- Shows target variable distribution
- Provides insights for modeling

**Run it:**
```bash
python data_exploration.py
```

## Key Findings

1. **Imbalanced Dataset**: 26.54% churn rate - may need balancing techniques
2. **Data Quality**: 11 non-numeric values in TotalCharges column need cleaning
3. **Contract Types**: Month-to-month contracts are most common (55%)
4. **Tenure**: Average customer tenure is 32 months
5. **Charges**: Monthly charges range from $18.25 to $118.75

## Next Steps

1. ✅ Dataset downloaded and explored
2. ✅ Comprehensive EDA with interactive Streamlit dashboard
3. ✅ Data preprocessing and cleaning
4. ✅ Feature engineering and encoding
5. ✅ Target variable balancing (undersampling)
6. Model selection and training
7. Model evaluation and comparison
8. Hyperparameter tuning
9. Final model deployment

## Project Structure
```
churn/
├── churn_venv/                          # Virtual environment
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
│   ├── X_train.csv                      # Training features (prepared)
│   ├── X_test.csv                       # Test features (prepared)
│   ├── y_train.csv                      # Training labels (prepared)
│   └── y_test.csv                       # Test labels (prepared)
├── artifacts/
│   ├── scaler.pkl                       # Fitted StandardScaler
│   └── feature_names.pkl                # Feature names list
├── data_exploration.py                  # Initial exploration script
├── eda_streamlit.py                     # Interactive EDA dashboard
├── eda_utils.py                         # EDA utility functions
├── data_preparation.py                  # Data preprocessing script
├── requirements.txt                     # Dependencies
└── README.md                            # This file
```

## Data Preparation Details

### Preprocessing Steps:
1. **Missing Values**: 11 missing values in TotalCharges filled with 0
2. **Feature Encoding**:
   - Binary columns: Partner, Dependents, PhoneService, PaperlessBilling
   - Ordinal encoding: Contract (Month-to-month → One year → Two year)
   - One-hot encoding: 9 nominal categorical features
3. **Target Balancing**: Naive undersampling (50/50 class distribution)
4. **Feature Scaling**: StandardScaler on numerical features
5. **Train/Test Split**: 80/20 split with stratification

### Prepared Dataset Statistics:
- **Training samples**: 2,990
- **Test samples**: 748
- **Total features**: 29
- **Class distribution**: Perfectly balanced (50/50)
- **Original data loss**: 3,305 samples (due to undersampling)

## Running the Project

### 1. Run EDA Dashboard
```bash
source churn_venv/bin/activate
streamlit run eda_streamlit.py
```
Open browser at http://localhost:8501

### 2. Run Data Preparation
```bash
python data_preparation.py
```

### 3. Train Models (Coming Soon)
```bash
python model_training.py
```
