#!/usr/bin/env python3
"""
Comprehensive analysis of the Milling Dataset
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Read the dataset
print("=" * 80)
print("MILLING DATASET ANALYSIS")
print("=" * 80)

# Load the data with semicolon delimiter and skip the first row (Column1, Column2, etc.)
df = pd.read_csv('FeatureAndMetadata_Milling.csv', sep=';', skiprows=1)

print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")
print(f"Dataset size: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

print("\n2. COLUMN INFORMATION")
print("-" * 80)
print(f"Total columns: {len(df.columns)}")
print("\nColumn categories:")

# Categorize columns
sensor_cols = [col for col in df.columns if any(x in col for x in ['Accelerometer', 'Current'])]
metadata_cols = [col for col in df.columns if col not in sensor_cols]

print(f"  - Sensor features: {len(sensor_cols)}")
print(f"  - Metadata features: {len(metadata_cols)}")

print("\nMetadata columns:")
for col in metadata_cols:
    print(f"  - {col}")

print("\n3. DATA TYPES")
print("-" * 80)
print(df.dtypes.value_counts())

print("\n4. MISSING VALUES")
print("-" * 80)
missing_data = df.isnull().sum()
if missing_data.sum() == 0:
    print("No missing values found!")
else:
    missing_percent = (missing_data / len(df) * 100).sort_values(ascending=False)
    missing_df = pd.DataFrame({
        'Missing Count': missing_data[missing_data > 0],
        'Percentage': missing_percent[missing_percent > 0]
    })
    print(missing_df)

print("\n5. BASIC STATISTICS - METADATA COLUMNS")
print("-" * 80)
print(df[metadata_cols].describe())

print("\n6. TARGET VARIABLE ANALYSIS (CycleToFailure)")
print("-" * 80)
if 'CycleToFailure' in df.columns:
    print(f"Mean cycles to failure: {df['CycleToFailure'].mean():.2f}")
    print(f"Median cycles to failure: {df['CycleToFailure'].median():.2f}")
    print(f"Min cycles to failure: {df['CycleToFailure'].min():.0f}")
    print(f"Max cycles to failure: {df['CycleToFailure'].max():.0f}")
    print(f"Std dev: {df['CycleToFailure'].std():.2f}")
    print(f"\nValue counts:")
    print(df['CycleToFailure'].value_counts().head(10))

print("\n7. CATEGORICAL FEATURES ANALYSIS")
print("-" * 80)

# Analyze categorical features
categorical_features = ['MillingToolType', 'ADOC', 'RDOC', 'ToolHolderLength']
for col in categorical_features:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Value counts:")
        print(df[col].value_counts().head(10).to_string(header=False).replace('\n', '\n    '))

print("\n8. SENSOR DATA STATISTICS (Summary)")
print("-" * 80)

# Group sensor columns by statistic type
stat_types = ['min', 'max', 'mean', 'std', 'skew', 'kurtosis']
for stat in stat_types:
    stat_cols = [col for col in sensor_cols if f' - {stat}' in col]
    if stat_cols:
        print(f"\n{stat.upper()} features: {len(stat_cols)} columns")
        print(f"  Overall mean: {df[stat_cols].mean().mean():.2f}")
        print(f"  Overall std: {df[stat_cols].std().mean():.2f}")

print("\n9. UNIQUE SAMPLES & CYCLES")
print("-" * 80)
if 'FileName' in df.columns:
    print(f"Unique files: {df['FileName'].nunique()}")
    print(f"\nSample file names:")
    print(df['FileName'].unique()[:10])

if 'NumberOfCycle' in df.columns:
    print(f"\nCycle range: {df['NumberOfCycle'].min()} to {df['NumberOfCycle'].max()}")
    print(f"Total unique cycles: {df['NumberOfCycle'].nunique()}")

print("\n10. DATA QUALITY CHECKS")
print("-" * 80)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Check for outliers in sensor data (using IQR method)
print("\nOutlier detection (IQR method) - Top 10 columns with outliers:")
outlier_counts = {}
for col in sensor_cols[:20]:  # Check first 20 sensor columns as sample
    if df[col].dtype in ['int64', 'float64']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            outlier_counts[col] = outliers

sorted_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:10]
for col, count in sorted_outliers:
    print(f"  {col}: {count} outliers ({count/len(df)*100:.2f}%)")

print("\n11. CORRELATION ANALYSIS (Metadata with Target)")
print("-" * 80)
if 'CycleToFailure' in df.columns:
    numeric_metadata = df[metadata_cols].select_dtypes(include=[np.number])
    if 'CycleToFailure' in numeric_metadata.columns:
        correlations = numeric_metadata.corr()['CycleToFailure'].drop('CycleToFailure').sort_values(ascending=False)
        print("Correlations with CycleToFailure:")
        print(correlations)

print("\n12. SAMPLE DATA")
print("-" * 80)
print("\nFirst 3 rows of metadata columns:")
print(df[metadata_cols].head(3))

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
