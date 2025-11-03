# Milling Dataset Analysis Report

## Executive Summary

This report provides a comprehensive analysis of the **FeatureAndMetadata_Milling.csv** dataset, which contains sensor data and metadata from milling machine operations. The dataset is designed for predictive maintenance tasks, particularly predicting tool failure cycles.

---

## 1. Dataset Overview

- **Total Records**: 968 samples
- **Total Features**: 131 columns
- **Dataset Size**: 1.17 MB
- **Missing Values**: None (100% complete dataset)
- **Duplicate Rows**: 0

### Data Composition

- **Sensor Features**: 120 columns (92% of features)
- **Metadata Features**: 11 columns (8% of features)

---

## 2. Feature Categories

### 2.1 Sensor Features (120 columns)

The sensor data is organized into **6 statistical aggregations** for each sensor:

1. **Minimum (min)** - 20 columns
2. **Maximum (max)** - 20 columns
3. **Mean (mean)** - 20 columns
4. **Standard Deviation (std)** - 20 columns
5. **Skewness (skew)** - 20 columns
6. **Kurtosis (kurtosis)** - 20 columns

#### Sensor Types:

**Accelerometers** (8 sensors):
- Spindle: +Y, -Z, -X directions
- X Driving axle: +Z, -X directions
- Y Driving axle: +Z, +Y, -X directions

**Current Sensors** (12 sensors):
- Spindle: L1, L2, L3 phases
- Driving axle X: L1, L2, L3 phases
- Driving axle Y: L1, L2, L3 phases
- Driving axle Z: L1, L2, L3 phases

### 2.2 Metadata Features (11 columns)

| Feature | Type | Description |
|---------|------|-------------|
| **FileName** | String | Unique identifier for each sample (e.g., P002_F01_C1) |
| **NumberOfCycle** | Integer | Current cycle number (1-150) |
| **SampleIndex** | Integer | Sample index within the dataset (2-119) |
| **TollIndex** | Integer | Tool index identifier |
| **MillingToolType** | Integer | Type of milling tool (1 or 2) |
| **ADOC** | Integer | Axial Depth of Cut (5 or 10) |
| **RDOC** | String | Radial Depth of Cut (4.5 or 8) |
| **HardnessMean** | Float | Material hardness (mean value) |
| **ToolHolderLength** | Integer | Length of tool holder (80 or 160 mm) |
| **CycleToFailure** | Integer | **TARGET VARIABLE** - Remaining cycles until tool failure |
| **CycleToFailureNormalized** | Float | Normalized version of CycleToFailure |

---

## 3. Target Variable Analysis: CycleToFailure

The primary prediction target is **CycleToFailure**, representing the remaining operational cycles before tool failure.

### Statistics:
- **Mean**: 48.52 cycles
- **Median**: 41.00 cycles
- **Standard Deviation**: 35.99 cycles
- **Range**: 0 to 149 cycles
- **Distribution**: Relatively uniform with slight peaks at early failure stages

### Key Insights:
- Tools fail across a wide range of cycles (0-149)
- The distribution suggests various failure modes depending on operating conditions
- Multiple samples (14) show immediate failure (CycleToFailure = 0)

---

## 4. Operating Conditions Analysis

### 4.1 Milling Tool Type
- **Type 1**: 617 samples (63.7%)
- **Type 2**: 351 samples (36.3%)

### 4.2 Axial Depth of Cut (ADOC)
- **5 mm**: 589 samples (60.8%)
- **10 mm**: 379 samples (39.2%)
- **Correlation with failure**: -0.167 (moderate negative)

### 4.3 Radial Depth of Cut (RDOC)
- **4.5 mm**: 756 samples (78.1%)
- **8 mm**: 212 samples (21.9%)

### 4.4 Tool Holder Length
- **80 mm**: 952 samples (98.3%)
- **160 mm**: 16 samples (1.7%)
- **Correlation with failure**: -0.151 (weak negative)

### 4.5 Material Hardness
- Material hardness values are recorded as mean hardness
- Distributed around typical industrial material hardness values

---

## 5. Correlation Analysis

### Correlations with CycleToFailure:

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| **NumberOfCycle** | -0.424 | **Strong negative** - As cycle number increases, remaining cycles decrease |
| **ADOC** | -0.167 | Moderate negative - Higher depth reduces tool life |
| **ToolHolderLength** | -0.151 | Weak negative - Longer holders slightly reduce tool life |
| **MillingToolType** | -0.036 | Very weak - Tool type has minimal direct impact |
| **SampleIndex** | -0.021 | Negligible |
| **TollIndex** | -0.018 | Negligible |

### Key Finding:
The **strongest predictor** of remaining tool life is the current cycle number, showing a strong inverse relationship (-0.424 correlation).

---

## 6. Sensor Data Statistics

### Average Values by Statistic Type:

| Statistic | Average Value | Std Dev |
|-----------|---------------|---------|
| **Min** | 7,722.95 | 9,891.86 |
| **Max** | 55,541.86 | 18,224.87 |
| **Mean** | 26,016.13 | 3,024.49 |
| **Std** | 8,548.16 | 3,882.39 |
| **Skew** | 0.42 | 0.84 |
| **Kurtosis** | 9.62 | 17.28 |

### Sensor Insights:
- High kurtosis values (9.62 average) indicate heavy-tailed distributions with outliers
- Positive skewness (0.42) suggests right-tailed distributions in sensor readings
- Current sensors show high variability compared to accelerometers

---

## 7. Data Quality Assessment

### 7.1 Completeness
✓ **Perfect**: No missing values across all 968 samples and 131 features

### 7.2 Uniqueness
✓ **Perfect**: No duplicate rows detected

### 7.3 Outliers
⚠ **Present**: Outliers detected in multiple sensor channels using IQR method:

**Top Outlier Sources**:
- Current - Driving axle Y (L1, L2, L3): 16-18% outliers
- Current - Driving axle X (L1, L2, L3): 12-13% outliers
- Accelerometer - Spindle sensors: 4-8% outliers

**Interpretation**:
- Outliers are likely genuine extreme operating conditions or transient events
- Should be retained for model training as they represent real-world variability

---

## 8. Key Findings and Recommendations

### 8.1 Dataset Strengths
1. **Complete Data**: No missing values, ready for immediate modeling
2. **Rich Features**: 120 statistical features from 20 sensors provide comprehensive coverage
3. **Balanced Coverage**: Good representation of different operating conditions
4. **Predictive Signal**: Clear correlation between cycle number and failure

### 8.2 Predictive Maintenance Opportunities
1. **Primary Target**: Predict CycleToFailure using sensor features
2. **Classification Task**: Binary classification (near failure vs. healthy)
3. **Regression Task**: Predict exact remaining cycles
4. **Anomaly Detection**: Identify unusual sensor patterns indicating impending failure

### 8.3 Recommended Modeling Approach

**Feature Engineering**:
- Consider interaction features between operating conditions (ADOC × RDOC)
- Time-series features from cycle progression
- Combine accelerometer and current sensor patterns

**Model Selection**:
- **Ensemble Methods**: Random Forest, Gradient Boosting (XGBoost, LightGBM)
- **Neural Networks**: For capturing complex sensor interactions
- **Time Series Models**: LSTM for sequential pattern recognition

**Validation Strategy**:
- Time-based split (train on early cycles, test on later cycles)
- Cross-validation by tool/file identifier to prevent data leakage

### 8.4 Business Applications
1. **Predictive Maintenance**: Schedule tool changes before failure
2. **Cost Optimization**: Maximize tool usage while preventing unexpected failures
3. **Process Optimization**: Identify optimal operating conditions for tool longevity
4. **Quality Control**: Detect anomalous operations early

---

## 9. Data Characteristics Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| **Completeness** | ✓ Excellent | No missing data |
| **Size** | ✓ Good | 968 samples sufficient for ML |
| **Feature Richness** | ✓ Excellent | 120 sensor features with statistical aggregations |
| **Target Distribution** | ✓ Good | Wide range of failure cycles |
| **Class Balance** | ⚠ Moderate | Some concentration at specific cycle values |
| **Data Quality** | ✓ Excellent | No duplicates, consistent formatting |
| **Outliers** | ⚠ Present | 4-18% in current sensors (likely genuine) |

---

## 10. Next Steps

1. **Model Development**:
   - Train baseline models (Random Forest, XGBoost)
   - Perform feature selection/importance analysis
   - Optimize hyperparameters

2. **Feature Analysis**:
   - Identify most predictive sensors for failure
   - Analyze sensor degradation patterns over cycles

3. **Validation**:
   - Test model performance on held-out tools
   - Evaluate prediction accuracy at different cycle stages

4. **Deployment**:
   - Develop real-time prediction system
   - Set up alerting thresholds for maintenance

---

## Conclusion

The **FeatureAndMetadata_Milling.csv** dataset is a high-quality, complete dataset well-suited for predictive maintenance modeling. With 120 sensor features and clear target variables, it provides excellent opportunities for developing accurate tool failure prediction models. The strong correlation between cycle number and remaining life, combined with rich sensor data, suggests that effective predictive models can be built to optimize maintenance schedules and reduce operational costs.

**Dataset Grade**: A (Excellent for machine learning)

---

*Report generated using comprehensive statistical analysis and exploratory data analysis techniques.*
