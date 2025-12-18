import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the extracted features
df = pd.read_csv(r'd:\Anomaly dataset\extracted_features.csv')

print("=" * 80)
print("FEATURE ANALYSIS FOR ANOMALY DETECTION MODEL TRAINING")
print("=" * 80)

print(f"\nDataFrame Shape: {df.shape}")
print(f"Total Features: {len(df.columns)}")

# 1. Feature Correlation Analysis
print("\n" + "=" * 80)
print("1. FEATURE CORRELATION ANALYSIS")
print("=" * 80)

# Select numeric features (exclude dataset, record_id, shaftSpeed, samplingRate)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['record_id', 'data_length']]

correlation_matrix = df[numeric_cols].corr()

# Find highly correlated features (> 0.9)
print("\nHighly Correlated Features (> 0.9):")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

if high_corr_pairs:
    for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {feat1:20s} <-> {feat2:20s} : {corr:.4f}")
else:
    print("  No highly correlated features found!")

# 2. Feature Variance Analysis
print("\n" + "=" * 80)
print("2. FEATURE VARIANCE & STANDARD DEVIATION")
print("=" * 80)

feature_stats = pd.DataFrame({
    'Feature': numeric_cols,
    'Mean': df[numeric_cols].mean(),
    'Std': df[numeric_cols].std(),
    'Var': df[numeric_cols].var(),
    'Min': df[numeric_cols].min(),
    'Max': df[numeric_cols].max(),
    'Range': df[numeric_cols].max() - df[numeric_cols].min()
})

feature_stats = feature_stats.sort_values('Var', ascending=False)
print(feature_stats.to_string())

# 3. RMS Analysis
print("\n" + "=" * 80)
print("3. RMS (ROOT MEAN SQUARE) ANALYSIS")
print("=" * 80)

rms_stats = df['rms'].describe()
print(f"\nRMS Statistics across all datasets:")
print(f"  Mean RMS: {df['rms'].mean():.6f}")
print(f"  Std RMS:  {df['rms'].std():.6f}")
print(f"  Min RMS:  {df['rms'].min():.6f}")
print(f"  Max RMS:  {df['rms'].max():.6f}")

print(f"\nRMS by Dataset:")
for dataset in sorted(df['dataset'].unique()):
    dataset_rms = df[df['dataset'] == dataset]['rms']
    print(f"  {dataset}: Mean={dataset_rms.mean():.6f}, Std={dataset_rms.std():.6f}, Range=[{dataset_rms.min():.6f}, {dataset_rms.max():.6f}]")

# 4. Time Domain Features 
print("\n" + "=" * 80)
print("4. RECOMMENDED TIME DOMAIN FEATURES FOR MODEL TRAINING")
print("=" * 80)

time_domain_features = ['mean', 'std', 'variance', 'rms', 'peak_to_peak', 
                        'crest_factor', 'skewness', 'kurtosis', 'iqr', 'median']
            
print("\nTop Time Domain Features (by variance):")
td_stats = feature_stats[feature_stats['Feature'].isin(time_domain_features)].head(10)
for idx, row in td_stats.iterrows():
    print(f"  {row['Feature']:20s} - Variance: {row['Var']:12.2f}, Range: [{row['Min']:10.4f}, {row['Max']:10.4f}]")

# 5. Envelope Features 
print("\n" + "=" * 80)
print("5. ENVELOPE FEATURES ANALYSIS")
print("=" * 80)

envelope_features = ['env_mean', 'env_std', 'env_max', 'env_min']
print("\nEnvelope Features:")
env_stats = feature_stats[feature_stats['Feature'].isin(envelope_features)]
for idx, row in env_stats.iterrows():
    print(f"  {row['Feature']:20s} - Variance: {row['Var']:12.2f}, Range: [{row['Min']:10.4f}, {row['Max']:10.4f}]")

# 6. Frequency Domain Features
print("\n" + "=" * 80)
print("6. FREQUENCY DOMAIN FEATURES ANALYSIS")
print("=" * 80)

freq_domain_features = ['total_power', 'power_mean', 'power_std', 'dominant_freq']
print("\nFrequency Domain Features:")
freq_stats = feature_stats[feature_stats['Feature'].isin(freq_domain_features)]
for idx, row in freq_stats.iterrows():
    print(f"  {row['Feature']:20s} - Variance: {row['Var']:12.2f}, Range: [{row['Min']:10.4f}, {row['Max']:10.4f}]")

# 7. Missing Values Check
print("\n" + "=" * 80)
print("7. MISSING VALUES CHECK")
print("=" * 80)

missing_vals = df[numeric_cols].isnull().sum()
print(f"\nTotal rows: {len(df)}")
if missing_vals.sum() == 0:
    print("✓ No missing values found!")
else:
    print("Missing values found:")
    print(missing_vals[missing_vals > 0])

# 8. Dataset Distribution
print("\n" + "=" * 80)
print("8. DATASET DISTRIBUTION")
print("=" * 80)

print("\nRecords per dataset:")
print(df['dataset'].value_counts().sort_index())

# 9. FEATURE RECOMMENDATIONS
print("\n" + "=" * 80)
print("9. RECOMMENDED FEATURES FOR ANOMALY DETECTION MODEL")
print("=" * 80)

print("""
PRIMARY FEATURES (Essential for model training):
  1. RMS (Root Mean Square) - Excellent feature for vibration analysis
  2. STD (Standard Deviation) - Measures signal variability
  3. Variance - Captures energy variations
  4. Crest Factor - Peak amplitude relative to RMS
  5. Peak-to-Peak - Dynamic range of signal

SECONDARY FEATURES (Enhanced discrimination):
  6. Skewness - Asymmetry of signal distribution
  7. Kurtosis - Tail behavior of distribution
  8. IQR (Interquartile Range) - Robust spread measure
  9. Env_Std (Envelope STD) - Envelope variation
 10. Env_Max - Peak envelope value

FREQUENCY DOMAIN FEATURES:
 11. Total_Power - Overall energy content
 12. Power_Mean - Average power spectrum
 13. Power_Std - Power spectrum variation
 14. Dominant_Freq - Primary frequency component

FEATURES TO CONSIDER REMOVING (Highly correlated):
  - Median (correlated with mean)
  - Min/Max (captured by peak_to_peak)
  - Env_Mean/Env_Min (less discriminative)
  - Mean (similar to RMS for this data)

OPTIMAL FEATURE SET FOR TRAINING:
  ['rms', 'std', 'variance', 'crest_factor', 'peak_to_peak', 
   'skewness', 'kurtosis', 'iqr', 'env_std', 'env_max',
   'total_power', 'power_std', 'dominant_freq']
   
Total: 13 features (balanced between interpretability and performance)
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
