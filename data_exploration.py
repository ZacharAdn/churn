"""
Telco Customer Churn Dataset - Exploration Script
================================================
This script explores the Telco Customer Churn dataset and explains its structure and features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("="*80)
print("TELCO CUSTOMER CHURN DATASET - EXPLORATORY ANALYSIS")
print("="*80)

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("\n" + "="*80)
print("1. BASIC DATASET INFORMATION")
print("="*80)

print(f"\nDataset Shape: {df.shape[0]} rows (customers) x {df.shape[1]} columns (features)")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "-"*80)
print("Column Names and Data Types:")
print("-"*80)
print(df.dtypes)

print("\n" + "="*80)
print("2. FIRST FEW ROWS OF DATA")
print("="*80)
print(df.head(10))

print("\n" + "="*80)
print("3. STATISTICAL SUMMARY")
print("="*80)
print(df.describe())

print("\n" + "="*80)
print("4. MISSING VALUES ANALYSIS")
print("="*80)
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': df.isnull().sum(),
    'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
if len(missing_data) > 0:
    print(missing_data)
else:
    print("No missing values found!")

print("\n" + "="*80)
print("5. TARGET VARIABLE ANALYSIS (Churn)")
print("="*80)
churn_counts = df['Churn'].value_counts()
churn_percentages = df['Churn'].value_counts(normalize=True) * 100

print("\nChurn Distribution:")
for value, count in churn_counts.items():
    percentage = churn_percentages[value]
    print(f"  {value}: {count} customers ({percentage:.2f}%)")

print("\n" + "="*80)
print("6. FEATURE CATEGORIES")
print("="*80)

# Categorize features
customer_info = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents']
service_features = ['tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
billing_features = ['Contract', 'PaperlessBilling', 'PaymentMethod', 
                   'MonthlyCharges', 'TotalCharges']

print("\n📊 CUSTOMER DEMOGRAPHIC INFORMATION:")
print("-" * 40)
for col in customer_info:
    if col in df.columns:
        if df[col].dtype == 'object':
            print(f"\n{col}:")
            print(df[col].value_counts())
        else:
            print(f"\n{col}: {df[col].nunique()} unique values")

print("\n\n📞 SERVICE FEATURES:")
print("-" * 40)
for col in service_features:
    if col in df.columns:
        if df[col].dtype == 'object':
            print(f"\n{col}:")
            print(df[col].value_counts())
        else:
            print(f"\n{col} (numerical):")
            print(f"  Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}")

print("\n\n💰 BILLING AND CONTRACT FEATURES:")
print("-" * 40)
for col in billing_features:
    if col in df.columns:
        if df[col].dtype == 'object' and col != 'TotalCharges':
            print(f"\n{col}:")
            print(df[col].value_counts())
        else:
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                print(f"\n{col}:")
                print(f"  Min: {numeric_col.min():.2f}, Max: {numeric_col.max():.2f}, Mean: {numeric_col.mean():.2f}")
            except:
                print(f"\n{col}: Data type issue")

print("\n" + "="*80)
print("7. KEY INSIGHTS")
print("="*80)

print("""
DATASET OVERVIEW:
-----------------
This dataset contains information about a telecommunications company's customers 
and whether they churned (left the company) or not.

FEATURE GROUPS:
---------------

1. CUSTOMER DEMOGRAPHICS (5 features):
   - customerID: Unique identifier for each customer
   - gender: Male/Female
   - SeniorCitizen: Whether customer is 65 or older (0/1)
   - Partner: Whether customer has a partner (Yes/No)
   - Dependents: Whether customer has dependents (Yes/No)

2. SERVICE FEATURES (11 features):
   - tenure: Number of months the customer has stayed with the company
   - PhoneService: Whether customer has phone service (Yes/No)
   - MultipleLines: Whether customer has multiple lines (Yes/No/No phone service)
   - InternetService: Type of internet service (DSL/Fiber optic/No)
   - OnlineSecurity: Whether customer has online security add-on (Yes/No/No internet)
   - OnlineBackup: Whether customer has online backup add-on (Yes/No/No internet)
   - DeviceProtection: Whether customer has device protection add-on (Yes/No/No internet)
   - TechSupport: Whether customer has tech support add-on (Yes/No/No internet)
   - StreamingTV: Whether customer has streaming TV add-on (Yes/No/No internet)
   - StreamingMovies: Whether customer has streaming movies add-on (Yes/No/No internet)

3. BILLING FEATURES (5 features):
   - Contract: Type of contract (Month-to-month/One year/Two year)
   - PaperlessBilling: Whether customer uses paperless billing (Yes/No)
   - PaymentMethod: Payment method (Electronic check/Mailed check/Bank transfer/Credit card)
   - MonthlyCharges: Monthly charge amount in dollars
   - TotalCharges: Total amount charged over tenure in dollars

4. TARGET VARIABLE:
   - Churn: Whether customer left the company (Yes/No)

POTENTIAL ANALYSIS DIRECTIONS:
------------------------------
1. Which services correlate most strongly with churn?
2. How does contract type affect customer retention?
3. What is the relationship between tenure and churn?
4. Do payment methods impact churn rates?
5. How do monthly charges relate to customer retention?
""")

print("\n" + "="*80)
print("8. DATA QUALITY ISSUES TO ADDRESS")
print("="*80)

# Check for potential data issues
issues = []

# Check TotalCharges
if 'TotalCharges' in df.columns:
    try:
        tc_numeric = pd.to_numeric(df['TotalCharges'], errors='coerce')
        non_numeric = df['TotalCharges'][tc_numeric.isna()]
        if len(non_numeric) > 0:
            issues.append(f"- TotalCharges has {len(non_numeric)} non-numeric values that need conversion")
    except:
        pass

# Check for duplicate customerIDs
if 'customerID' in df.columns:
    duplicates = df['customerID'].duplicated().sum()
    if duplicates > 0:
        issues.append(f"- Found {duplicates} duplicate customer IDs")

# Check for inconsistencies
if len(issues) > 0:
    print("\nIssues found:")
    for issue in issues:
        print(issue)
else:
    print("\nNo major data quality issues detected!")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
