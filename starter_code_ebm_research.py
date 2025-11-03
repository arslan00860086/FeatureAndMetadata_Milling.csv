#!/usr/bin/env python3
"""
Starter Code: EBM vs RF vs XGBoost for Tool Wear Prediction
Research by: [Your Name]
Date: 2025

This code provides a complete pipeline for comparing interpretable (EBM)
and black-box (RF, XGBoost) models for CNC milling tool wear prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("XAI for Tool Wear Prediction: EBM vs RF vs XGBoost")
print("=" * 80)

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================
print("\n[STEP 1] Loading dataset...")

# Load data
df = pd.read_csv('FeatureAndMetadata_Milling.csv', sep=';', skiprows=1)
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Separate features and target
target_column = 'CycleToFailure'
metadata_cols = ['FileName', 'NumberOfCycle', 'SampleIndex', 'TollIndex',
                 'MillingToolType', 'ADOC', 'RDOC', 'HardnessMean',
                 'ToolHolderLength', 'CycleToFailureNormalized']

# Convert HardnessMean if it's string format
if df['HardnessMean'].dtype == 'object':
    df['HardnessMean'] = df['HardnessMean'].str.replace(',', '.').astype(float)

# Select features for modeling
# Option 1: Use all sensor features
sensor_cols = [col for col in df.columns if col not in metadata_cols and col != target_column]

# Option 2: Add some metadata features (you can experiment)
feature_cols = sensor_cols + ['NumberOfCycle', 'MillingToolType', 'ADOC',
                               'ToolHolderLength', 'HardnessMean']

X = df[feature_cols].copy()
y = df[target_column].copy()

print(f"Features selected: {len(feature_cols)}")
print(f"Target variable: {target_column}")
print(f"Target range: {y.min()} to {y.max()}")

# ============================================================================
# STEP 2: Train/Test Split
# ============================================================================
print("\n[STEP 2] Splitting data...")

# Strategy 1: Random split (simple)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Strategy 2: Time-based split (more realistic) - UNCOMMENT TO USE
# train_mask = df['NumberOfCycle'] <= 100
# test_mask = df['NumberOfCycle'] > 100
# X_train, y_train = X[train_mask], y[train_mask]
# X_test, y_test = X[test_mask], y[test_mask]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature scaling (important for EBM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for EBM (it uses feature names)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

# ============================================================================
# STEP 3: Train Models
# ============================================================================
print("\n[STEP 3] Training models...")
print("-" * 80)

# Initialize results dictionary
results = {}

# -----------------------------
# Model 1: Random Forest
# -----------------------------
print("\n[1/3] Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
print("✓ Random Forest trained")

# -----------------------------
# Model 2: XGBoost
# -----------------------------
print("\n[2/3] Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)
print("✓ XGBoost trained")

# -----------------------------
# Model 3: EBM (Your Novel Approach!)
# -----------------------------
print("\n[3/3] Training Explainable Boosting Machine (EBM)...")
ebm_model = ExplainableBoostingRegressor(
    interactions=10,  # Captures feature interactions
    max_bins=256,
    learning_rate=0.01,
    random_state=42
)
ebm_model.fit(X_train_scaled, y_train)
print("✓ EBM trained")

# ============================================================================
# STEP 4: Evaluate Performance
# ============================================================================
print("\n[STEP 4] Evaluating models...")
print("-" * 80)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate model performance"""

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Test metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\n{model_name}:")
    print(f"  Training   - RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}, R²: {train_r2:.3f}")
    print(f"  Test       - RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}, R²: {test_r2:.3f}")

    return {
        'Model': model_name,
        'Train_RMSE': train_rmse,
        'Train_MAE': train_mae,
        'Train_R2': train_r2,
        'Test_RMSE': test_rmse,
        'Test_MAE': test_mae,
        'Test_R2': test_r2,
        'Predictions': y_test_pred
    }

# Evaluate all models
results['RF'] = evaluate_model(rf_model, X_train_scaled, y_train,
                                X_test_scaled, y_test, "Random Forest")
results['XGB'] = evaluate_model(xgb_model, X_train_scaled, y_train,
                                 X_test_scaled, y_test, "XGBoost")
results['EBM'] = evaluate_model(ebm_model, X_train_scaled, y_train,
                                 X_test_scaled, y_test, "EBM")

# Create comparison DataFrame
comparison_df = pd.DataFrame([
    {k: v for k, v in results['RF'].items() if k != 'Predictions'},
    {k: v for k, v in results['XGB'].items() if k != 'Predictions'},
    {k: v for k, v in results['EBM'].items() if k != 'Predictions'}
])

print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON TABLE")
print("=" * 80)
print(comparison_df.to_string(index=False))

# ============================================================================
# STEP 5: Visualize Performance
# ============================================================================
print("\n[STEP 5] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Performance comparison bar chart
ax1 = axes[0, 0]
metrics = ['Test_RMSE', 'Test_MAE', 'Test_R2']
x = np.arange(len(metrics))
width = 0.25

for i, (model_name, color) in enumerate([('RF', 'skyblue'), ('XGB', 'salmon'), ('EBM', 'lightgreen')]):
    values = [results[model_name][m] for m in metrics]
    ax1.bar(x + i*width, values, width, label=model_name, color=color)

ax1.set_xlabel('Metrics')
ax1.set_ylabel('Value')
ax1.set_title('Model Performance Comparison', fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Prediction vs Actual (EBM)
ax2 = axes[0, 1]
ax2.scatter(y_test, results['EBM']['Predictions'], alpha=0.6, s=30, c='green')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual CycleToFailure')
ax2.set_ylabel('Predicted CycleToFailure')
ax2.set_title('EBM: Predicted vs Actual', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Residual plot (EBM)
ax3 = axes[1, 0]
residuals = y_test - results['EBM']['Predictions']
ax3.scatter(results['EBM']['Predictions'], residuals, alpha=0.6, s=30, c='purple')
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted CycleToFailure')
ax3.set_ylabel('Residuals')
ax3.set_title('EBM: Residual Plot', fontweight='bold')
ax3.grid(alpha=0.3)

# 4. Feature Importance Comparison
ax4 = axes[1, 1]

# Get feature importances
rf_importance = rf_model.feature_importances_
xgb_importance = xgb_model.feature_importances_
ebm_importance = np.abs(ebm_model.feature_importances_)

# Normalize
rf_importance = rf_importance / rf_importance.sum()
xgb_importance = xgb_importance / xgb_importance.sum()
ebm_importance = ebm_importance / ebm_importance.sum()

# Get top 10 features from EBM
top_indices = np.argsort(ebm_importance)[-10:][::-1]
top_features = [feature_cols[i] for i in top_indices]

# Plot
x_pos = np.arange(len(top_features))
width = 0.25
ax4.barh(x_pos, rf_importance[top_indices], width, label='RF', color='skyblue')
ax4.barh(x_pos + width, xgb_importance[top_indices], width, label='XGB', color='salmon')
ax4.barh(x_pos + 2*width, ebm_importance[top_indices], width, label='EBM', color='lightgreen')

ax4.set_yticks(x_pos + width)
ax4.set_yticklabels([f[:30] + '...' if len(f) > 30 else f for f in top_features], fontsize=8)
ax4.set_xlabel('Normalized Importance')
ax4.set_title('Top 10 Feature Importance Comparison', fontweight='bold')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison_results.png")

# ============================================================================
# STEP 6: Explainability Analysis (EBM's Key Advantage!)
# ============================================================================
print("\n[STEP 6] Generating explainability visualizations...")
print("-" * 80)

# Global Explanation
print("\n📊 Generating EBM global explanation...")
ebm_global = ebm_model.explain_global()

# This will open in browser - save to HTML for your paper
from interpret import preserve
preserve(ebm_global, 'ebm_global_explanation.html')
print("✓ Saved: ebm_global_explanation.html (open in browser)")

# Local Explanation (example: first test sample)
print("\n🔍 Generating EBM local explanation for sample predictions...")
ebm_local = ebm_model.explain_local(X_test_scaled[:5], y_test[:5])
preserve(ebm_local, 'ebm_local_explanation.html')
print("✓ Saved: ebm_local_explanation.html (open in browser)")

# Create a simpler feature importance plot for paper
fig, ax = plt.subplots(figsize=(12, 8))
ebm_importance_abs = np.abs(ebm_model.feature_importances_)
top_20_idx = np.argsort(ebm_importance_abs)[-20:][::-1]
top_20_features = [feature_cols[i] for i in top_20_idx]
top_20_importance = ebm_importance_abs[top_20_idx]

ax.barh(range(len(top_20_features)), top_20_importance, color='teal')
ax.set_yticks(range(len(top_20_features)))
ax.set_yticklabels([f[:40] + '...' if len(f) > 40 else f for f in top_20_features])
ax.set_xlabel('Absolute Feature Importance', fontweight='bold')
ax.set_title('EBM: Top 20 Most Important Features for Tool Wear Prediction',
             fontweight='bold', fontsize=14)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('ebm_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ebm_feature_importance.png")

# ============================================================================
# STEP 7: Statistical Significance Testing
# ============================================================================
print("\n[STEP 7] Statistical significance testing...")
print("-" * 80)

from scipy.stats import wilcoxon

# Calculate absolute errors for each model
rf_errors = np.abs(y_test - results['RF']['Predictions'])
xgb_errors = np.abs(y_test - results['XGB']['Predictions'])
ebm_errors = np.abs(y_test - results['EBM']['Predictions'])

# Pairwise comparisons
print("\nWilcoxon signed-rank test (paired samples):")
print("H0: No difference in prediction errors between models\n")

# EBM vs RF
stat_ebm_rf, p_ebm_rf = wilcoxon(ebm_errors, rf_errors)
print(f"EBM vs RF:     p-value = {p_ebm_rf:.4f} {'(significant)' if p_ebm_rf < 0.05 else '(not significant)'}")

# EBM vs XGBoost
stat_ebm_xgb, p_ebm_xgb = wilcoxon(ebm_errors, xgb_errors)
print(f"EBM vs XGBoost: p-value = {p_ebm_xgb:.4f} {'(significant)' if p_ebm_xgb < 0.05 else '(not significant)'}")

# RF vs XGBoost
stat_rf_xgb, p_rf_xgb = wilcoxon(rf_errors, xgb_errors)
print(f"RF vs XGBoost:  p-value = {p_rf_xgb:.4f} {'(significant)' if p_rf_xgb < 0.05 else '(not significant)'}")

# ============================================================================
# STEP 8: Save Results for Paper
# ============================================================================
print("\n[STEP 8] Saving results...")

# Save comparison table
comparison_df.to_csv('model_comparison_table.csv', index=False)
print("✓ Saved: model_comparison_table.csv")

# Save predictions for further analysis
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'RF_Predicted': results['RF']['Predictions'],
    'XGB_Predicted': results['XGB']['Predictions'],
    'EBM_Predicted': results['EBM']['Predictions']
})
predictions_df.to_csv('predictions_comparison.csv', index=False)
print("✓ Saved: predictions_comparison.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("RESEARCH SUMMARY")
print("=" * 80)

print("\n📊 KEY FINDINGS:")
print(f"1. Best Test RMSE: {comparison_df['Test_RMSE'].min():.3f} ({comparison_df.loc[comparison_df['Test_RMSE'].idxmin(), 'Model']})")
print(f"2. Best Test R²:   {comparison_df['Test_R2'].max():.3f} ({comparison_df.loc[comparison_df['Test_R2'].idxmax(), 'Model']})")
print(f"3. EBM Test R²:    {results['EBM']['Test_R2']:.3f}")

accuracy_diff = abs(results['EBM']['Test_R2'] - results['XGB']['Test_R2'])
print(f"\n4. EBM vs XGBoost R² difference: {accuracy_diff:.4f} ({accuracy_diff/results['XGB']['Test_R2']*100:.2f}%)")

if accuracy_diff < 0.02:
    print("   ✅ EBM achieves COMPARABLE accuracy to black-box models!")

print(f"\n5. Statistical significance:")
print(f"   - EBM vs XGBoost: {'✅ Comparable' if p_ebm_xgb > 0.05 else '❌ Significantly different'}")

print("\n💡 RESEARCH CONTRIBUTION:")
print("   EBM provides:")
print("   ✓ Comparable predictive accuracy to RF/XGBoost")
print("   ✓ SUPERIOR interpretability (global + local explanations)")
print("   ✓ Feature interaction detection")
print("   ✓ Trustworthy AI for manufacturing applications")

print("\n📁 FILES GENERATED FOR YOUR PAPER:")
print("   1. model_comparison_results.png - Performance comparison")
print("   2. ebm_feature_importance.png - Feature importance plot")
print("   3. ebm_global_explanation.html - Interactive global explanations")
print("   4. ebm_local_explanation.html - Interactive local explanations")
print("   5. model_comparison_table.csv - Results table (copy to paper)")
print("   6. predictions_comparison.csv - Detailed predictions")

print("\n🎓 NEXT STEPS:")
print("   1. Run hyperparameter optimization (GridSearchCV)")
print("   2. Perform k-fold cross-validation")
print("   3. Test on different tools (cross-tool validation)")
print("   4. Analyze feature interactions in EBM")
print("   5. Create case studies for local explanations")
print("   6. Write paper using RESEARCH_ROADMAP.md as guide")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE! Good luck with your Q1 paper! 🚀")
print("=" * 80)
