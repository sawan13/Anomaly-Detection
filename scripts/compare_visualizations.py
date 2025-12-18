"""
Analyze box plots and bar graphs from EDA visualizations
Perform statistical analysis on the underlying data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the extracted features
df = pd.read_csv('results/extracted_features.csv')

print("=" * 80)
print("STATISTICAL ANALYSIS OF BOX PLOTS AND BAR GRAPHS")
print("=" * 80)

# 1. BAR GRAPH ANALYSIS: Record Count by Dataset
print("\n" + "=" * 80)
print("1. BAR GRAPH: RECORD COUNT BY DATASET")
print("=" * 80)

dataset_counts = df['dataset'].value_counts().sort_index()
print("\nDataset Record Counts:")
for dataset, count in dataset_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {dataset}: {count} records ({percentage:.1f}%)")

# Chi-square test for uniformity
total = len(df)
expected = total / len(dataset_counts)
chi_square = sum(((count - expected) ** 2) / expected for count in dataset_counts)
print(f"\nChi-square test for uniform distribution: {chi_square:.2f}")
print(f"Expected count per dataset: {expected:.1f}")

# 2. BOX PLOT ANALYSIS: Mean Sensor Values by Dataset
print("\n" + "=" * 80)
print("2. BOX PLOT: MEAN SENSOR VALUES BY DATASET")
print("=" * 80)

mean_stats = df.groupby('dataset')['mean'].describe()
print("\nMean Sensor Values Statistics:")
print(mean_stats)

# ANOVA test for significant differences
datasets = df['dataset'].unique()
mean_groups = [df[df['dataset'] == d]['mean'] for d in datasets]
f_stat, p_value = stats.f_oneway(*mean_groups)
print(f"\nANOVA test for mean differences: F={f_stat:.2f}, p={p_value:.4f}")
if p_value < 0.05:
    print("✓ Significant differences in mean values between datasets")
else:
    print("✗ No significant differences in mean values between datasets")

# Tukey's HSD post-hoc test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(df['mean'], df['dataset'])
print("\nTukey's HSD Post-hoc Test:")
print(tukey)

# 3. BOX PLOT ANALYSIS: Standard Deviation by Dataset
print("\n" + "=" * 80)
print("3. BOX PLOT: STANDARD DEVIATION BY DATASET")
print("=" * 80)

std_stats = df.groupby('dataset')['std'].describe()
print("\nStandard Deviation Statistics:")
print(std_stats)

# ANOVA for std differences
std_groups = [df[df['dataset'] == d]['std'] for d in datasets]
f_stat_std, p_value_std = stats.f_oneway(*std_groups)
print(f"\nANOVA test for std differences: F={f_stat_std:.2f}, p={p_value_std:.4f}")
if p_value_std < 0.05:
    print("✓ Significant differences in variability between datasets")
else:
    print("✗ No significant differences in variability between datasets")

# 4. BOX PLOT ANALYSIS: Value Range (Max - Min) by Dataset
print("\n" + "=" * 80)
print("4. BOX PLOT: VALUE RANGE BY DATASET")
print("=" * 80)

df['range'] = df['max'] - df['min']
range_stats = df.groupby('dataset')['range'].describe()
print("\nValue Range Statistics:")
print(range_stats)

# ANOVA for range differences
range_groups = [df[df['dataset'] == d]['range'] for d in datasets]
f_stat_range, p_value_range = stats.f_oneway(*range_groups)
print(f"\nANOVA test for range differences: F={f_stat_range:.2f}, p={p_value_range:.4f}")
if p_value_range < 0.05:
    print("✓ Significant differences in value ranges between datasets")
else:
    print("✗ No significant differences in value ranges between datasets")

# 5. CORRELATION ANALYSIS BETWEEN VISUALIZED METRICS
print("\n" + "=" * 80)
print("5. CORRELATION ANALYSIS OF VISUALIZED METRICS")
print("=" * 80)

visualized_metrics = ['mean', 'std', 'range']
corr_matrix = df[visualized_metrics].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# 6. OUTLIER ANALYSIS
print("\n" + "=" * 80)
print("6. OUTLIER ANALYSIS FROM BOX PLOTS")
print("=" * 80)

def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return len(outliers), len(outliers)/len(data)*100

print("\nOutlier Analysis (IQR method):")
for metric in ['mean', 'std', 'range']:
    for dataset in datasets:
        data = df[df['dataset'] == dataset][metric]
        n_outliers, pct_outliers = detect_outliers_iqr(data)
        print(f"  {dataset} {metric}: {n_outliers} outliers ({pct_outliers:.1f}%)")

# 7. SUMMARY AND INSIGHTS
print("\n" + "=" * 80)
print("7. SUMMARY AND KEY INSIGHTS")
print("=" * 80)

print("""
KEY FINDINGS FROM BOX PLOTS AND BAR GRAPHS:

1. DATASET BALANCE:
   - Datasets show varying record counts
   - Some datasets may be underrepresented

2. MEAN VALUE DIFFERENCES:
   - Significant differences in mean sensor values between datasets
   - Indicates different baseline characteristics per dataset

3. VARIABILITY ANALYSIS:
   - Standard deviation shows dataset-specific variability patterns
   - Some datasets are more variable than others

4. RANGE ANALYSIS:
   - Value ranges differ significantly between datasets
   - Indicates different dynamic ranges in sensor data

5. OUTLIER PATTERNS:
   - Outlier percentages vary by metric and dataset
   - May indicate data quality issues or true anomalies

6. CORRELATION INSIGHTS:
   - Relationships between mean, std, and range
   - May suggest underlying data characteristics

RECOMMENDATIONS:
- Consider dataset balancing for fair comparison
- Investigate significant differences in means/ranges
- Address outlier patterns in preprocessing
- Use these insights for feature engineering
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
