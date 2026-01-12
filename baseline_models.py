"""
Baseline Model Training for Churn Prediction
=============================================
Train and evaluate baseline machine learning models:
- Logistic Regression
- Decision Tree

These will serve as baseline models to compare against more complex algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import pickle
import json
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BASELINE MODEL TRAINING - CHURN PREDICTION")
print("="*80)

# =============================================================================
# STEP 1: LOAD PREPARED DATA
# =============================================================================

print("\n[STEP 1] Loading prepared data...")

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")
print(f"✓ Class distribution in training: {np.bincount(y_train)}")
print(f"✓ Class distribution in test: {np.bincount(y_test)}")

# =============================================================================
# STEP 2: TRAIN LOGISTIC REGRESSION
# =============================================================================

print("\n" + "="*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*80)

print("\n[Training] Fitting Logistic Regression model...")
start_time = time.time()

# Create and train model
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='liblinear'
)
lr_model.fit(X_train, y_train)

training_time_lr = time.time() - start_time
print(f"✓ Training completed in {training_time_lr:.2f} seconds")

# Make predictions
print("[Evaluation] Making predictions on test set...")
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Calculate metrics
lr_metrics = {
    'model_name': 'Logistic Regression',
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1_score': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_lr),
    'training_time': training_time_lr,
    'confusion_matrix': confusion_matrix(y_test, y_pred_lr).tolist()
}

print("\n✓ Logistic Regression Results:")
print(f"  Accuracy:  {lr_metrics['accuracy']:.4f}")
print(f"  Precision: {lr_metrics['precision']:.4f}")
print(f"  Recall:    {lr_metrics['recall']:.4f}")
print(f"  F1-Score:  {lr_metrics['f1_score']:.4f}")
print(f"  ROC-AUC:   {lr_metrics['roc_auc']:.4f}")

print("\n  Confusion Matrix:")
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f"    [[TN={cm_lr[0,0]}  FP={cm_lr[0,1]}]")
print(f"     [FN={cm_lr[1,0]}  TP={cm_lr[1,1]}]]")

# Save model
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("\n✓ Model saved: logistic_regression_model.pkl")

# =============================================================================
# STEP 3: TRAIN DECISION TREE
# =============================================================================

print("\n" + "="*80)
print("MODEL 2: DECISION TREE")
print("="*80)

print("\n[Training] Fitting Decision Tree model...")
start_time = time.time()

# Create and train model
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
dt_model.fit(X_train, y_train)

training_time_dt = time.time() - start_time
print(f"✓ Training completed in {training_time_dt:.2f} seconds")

# Make predictions
print("[Evaluation] Making predictions on test set...")
y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]

# Calculate metrics
dt_metrics = {
    'model_name': 'Decision Tree',
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'precision': precision_score(y_test, y_pred_dt),
    'recall': recall_score(y_test, y_pred_dt),
    'f1_score': f1_score(y_test, y_pred_dt),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_dt),
    'training_time': training_time_dt,
    'confusion_matrix': confusion_matrix(y_test, y_pred_dt).tolist()
}

print("\n✓ Decision Tree Results:")
print(f"  Accuracy:  {dt_metrics['accuracy']:.4f}")
print(f"  Precision: {dt_metrics['precision']:.4f}")
print(f"  Recall:    {dt_metrics['recall']:.4f}")
print(f"  F1-Score:  {dt_metrics['f1_score']:.4f}")
print(f"  ROC-AUC:   {dt_metrics['roc_auc']:.4f}")

print("\n  Confusion Matrix:")
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(f"    [[TN={cm_dt[0,0]}  FP={cm_dt[0,1]}]")
print(f"     [FN={cm_dt[1,0]}  TP={cm_dt[1,1]}]]")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Top 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# Save model
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)
print("\n✓ Model saved: decision_tree_model.pkl")

# Save feature importance
feature_importance.to_csv('feature_importance_dt.csv', index=False)
print("✓ Feature importance saved: feature_importance_dt.csv")

# =============================================================================
# STEP 4: COMPARE MODELS
# =============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame([lr_metrics, dt_metrics])
comparison = comparison[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'training_time']]

print("\n")
print(comparison.to_string(index=False))

# Determine best model
best_model_idx = comparison['f1_score'].idxmax()
best_model = comparison.iloc[best_model_idx]['model_name']

print(f"\n🏆 Best Model (by F1-Score): {best_model}")
print(f"   F1-Score: {comparison.iloc[best_model_idx]['f1_score']:.4f}")

# =============================================================================
# STEP 5: SAVE RESULTS
# =============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save all metrics to JSON
results = {
    'logistic_regression': lr_metrics,
    'decision_tree': dt_metrics,
    'best_model': best_model,
    'comparison': comparison.to_dict('records')
}

with open('baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved: baseline_results.json")

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test,
    'lr_prediction': y_pred_lr,
    'lr_probability': y_pred_proba_lr,
    'dt_prediction': y_pred_dt,
    'dt_probability': y_pred_proba_dt
})
predictions_df.to_csv('baseline_predictions.csv', index=False)
print("✓ Predictions saved: baseline_predictions.csv")

# =============================================================================
# STEP 6: SUMMARY
# =============================================================================

print("\n" + "="*80)
print("BASELINE MODELS TRAINING SUMMARY")
print("="*80)

print(f"""
✓ Two baseline models trained and evaluated:
  1. Logistic Regression
  2. Decision Tree

✓ Performance Metrics:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Confusion matrices computed

✓ Models Saved:
  - logistic_regression_model.pkl
  - decision_tree_model.pkl

✓ Results Saved:
  - baseline_results.json (all metrics)
  - baseline_predictions.csv (test predictions)
  - feature_importance_dt.csv (Decision Tree feature importance)

Best Model: {best_model}
F1-Score: {comparison.iloc[best_model_idx]['f1_score']:.4f}

These baseline models provide a starting point for comparison with
more advanced algorithms (Random Forest, XGBoost, etc.)
""")

print("="*80)
print("✓ BASELINE MODEL TRAINING COMPLETE!")
print("="*80)
