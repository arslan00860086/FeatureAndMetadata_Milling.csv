#!/usr/bin/env python3
"""
Create visualizations for the Milling Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('FeatureAndMetadata_Milling.csv', sep=';', skiprows=1)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 16))

# 1. Distribution of CycleToFailure
ax1 = plt.subplot(3, 3, 1)
df['CycleToFailure'].hist(bins=50, ax=ax1, edgecolor='black')
ax1.set_title('Distribution of Cycles to Failure', fontsize=12, fontweight='bold')
ax1.set_xlabel('Cycles to Failure')
ax1.set_ylabel('Frequency')
ax1.axvline(df['CycleToFailure'].mean(), color='red', linestyle='--', label=f'Mean: {df["CycleToFailure"].mean():.1f}')
ax1.axvline(df['CycleToFailure'].median(), color='green', linestyle='--', label=f'Median: {df["CycleToFailure"].median():.1f}')
ax1.legend()

# 2. CycleToFailure by MillingToolType
ax2 = plt.subplot(3, 3, 2)
df.groupby('MillingToolType')['CycleToFailure'].mean().plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black')
ax2.set_title('Average Cycles to Failure by Tool Type', fontsize=12, fontweight='bold')
ax2.set_xlabel('Milling Tool Type')
ax2.set_ylabel('Avg Cycles to Failure')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

# 3. CycleToFailure by ADOC
ax3 = plt.subplot(3, 3, 3)
df.groupby('ADOC')['CycleToFailure'].mean().plot(kind='bar', ax=ax3, color='lightcoral', edgecolor='black')
ax3.set_title('Average Cycles to Failure by ADOC', fontsize=12, fontweight='bold')
ax3.set_xlabel('Axial Depth of Cut (ADOC)')
ax3.set_ylabel('Avg Cycles to Failure')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

# 4. CycleToFailure by RDOC
ax4 = plt.subplot(3, 3, 4)
rdoc_counts = df.groupby('RDOC')['CycleToFailure'].mean()
ax4.bar(range(len(rdoc_counts)), rdoc_counts.values, color='lightgreen', edgecolor='black')
ax4.set_title('Average Cycles to Failure by RDOC', fontsize=12, fontweight='bold')
ax4.set_xlabel('Radial Depth of Cut (RDOC)')
ax4.set_ylabel('Avg Cycles to Failure')
ax4.set_xticks(range(len(rdoc_counts)))
ax4.set_xticklabels(rdoc_counts.index, rotation=0)

# 5. CycleToFailure vs NumberOfCycle
ax5 = plt.subplot(3, 3, 5)
ax5.scatter(df['NumberOfCycle'], df['CycleToFailure'], alpha=0.5, s=20)
ax5.set_title('Cycles to Failure vs Number of Cycle', fontsize=12, fontweight='bold')
ax5.set_xlabel('Number of Cycle')
ax5.set_ylabel('Cycles to Failure')
z = np.polyfit(df['NumberOfCycle'], df['CycleToFailure'], 1)
p = np.poly1d(z)
ax5.plot(df['NumberOfCycle'], p(df['NumberOfCycle']), "r--", alpha=0.8, linewidth=2)

# 6. Correlation heatmap for metadata
ax6 = plt.subplot(3, 3, 6)
numeric_metadata = df[['NumberOfCycle', 'SampleIndex', 'TollIndex', 'MillingToolType',
                         'ADOC', 'ToolHolderLength', 'CycleToFailure']].corr()
sns.heatmap(numeric_metadata, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax6,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax6.set_title('Correlation Matrix - Metadata Features', fontsize=12, fontweight='bold')

# 7. Distribution of HardnessMean
ax7 = plt.subplot(3, 3, 7)
# Convert HardnessMean to numeric, replacing commas with periods
df['HardnessMean_numeric'] = df['HardnessMean'].str.replace(',', '.').astype(float)
df['HardnessMean_numeric'].hist(bins=30, ax=ax7, edgecolor='black', color='gold')
ax7.set_title('Distribution of Material Hardness', fontsize=12, fontweight='bold')
ax7.set_xlabel('Hardness Mean')
ax7.set_ylabel('Frequency')

# 8. Tool Holder Length Distribution
ax8 = plt.subplot(3, 3, 8)
df['ToolHolderLength'].value_counts().plot(kind='bar', ax=ax8, color='mediumpurple', edgecolor='black')
ax8.set_title('Tool Holder Length Distribution', fontsize=12, fontweight='bold')
ax8.set_xlabel('Tool Holder Length')
ax8.set_ylabel('Count')
ax8.set_xticklabels(ax8.get_xticklabels(), rotation=0)

# 9. Sample sensor statistics
ax9 = plt.subplot(3, 3, 9)
# Get mean values for different sensor statistics
sensor_stats = {
    'Min': df[[col for col in df.columns if ' - min' in col]].mean().mean(),
    'Max': df[[col for col in df.columns if ' - max' in col]].mean().mean(),
    'Mean': df[[col for col in df.columns if ' - mean' in col]].mean().mean(),
    'Std': df[[col for col in df.columns if ' - std' in col]].mean().mean()
}
ax9.bar(sensor_stats.keys(), sensor_stats.values(), color=['red', 'blue', 'green', 'orange'],
        edgecolor='black', alpha=0.7)
ax9.set_title('Average Sensor Readings by Statistic Type', fontsize=12, fontweight='bold')
ax9.set_xlabel('Statistic Type')
ax9.set_ylabel('Average Value')
ax9.set_yscale('log')

plt.tight_layout()
plt.savefig('milling_dataset_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'milling_dataset_analysis.png'")

# Create additional detailed visualizations

# Sensor analysis
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accelerometer analysis
ax1 = axes[0, 0]
accel_cols = [col for col in df.columns if 'Accelerometer' in col and ' - mean' in col]
df[accel_cols].mean().plot(kind='barh', ax=ax1, color='teal')
ax1.set_title('Average Accelerometer Readings (Mean)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Average Value')

# Current analysis
ax2 = axes[0, 1]
current_cols = [col for col in df.columns if 'Current' in col and ' - mean' in col]
df[current_cols].mean().plot(kind='barh', ax=ax2, color='orangered')
ax2.set_title('Average Current Readings (Mean)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Average Value')

# Skewness distribution
ax3 = axes[1, 0]
skew_cols = [col for col in df.columns if ' - skew' in col]
df[skew_cols].mean().hist(bins=20, ax=ax3, edgecolor='black', color='lightblue')
ax3.set_title('Distribution of Sensor Skewness Values', fontsize=12, fontweight='bold')
ax3.set_xlabel('Skewness')
ax3.set_ylabel('Frequency')

# Kurtosis distribution
ax4 = axes[1, 1]
kurt_cols = [col for col in df.columns if ' - kurtosis' in col]
df[kurt_cols].mean().hist(bins=20, ax=ax4, edgecolor='black', color='salmon')
ax4.set_title('Distribution of Sensor Kurtosis Values', fontsize=12, fontweight='bold')
ax4.set_xlabel('Kurtosis')
ax4.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('milling_sensor_details.png', dpi=300, bbox_inches='tight')
print("Sensor visualization saved as 'milling_sensor_details.png'")

print("\nVisualization creation complete!")
