"""
Bar Graph Comparison for 8 Time Domain Features
Compares: rms, skewness, peak_to_peak, median, kurtosis, iqr, crest_factor, variance
Across datasets (AGR.361, AGR.561, BIS.361, BSK.362) and normal vs anomaly
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
features_csv = 'results/extracted_features.csv'
results_csv = 'results/reconstruction_results.csv'

df_features = pd.read_csv(features_csv)
df_results = pd.read_csv(results_csv)

# Merge features with anomaly labels
# Both files have the same records in same order, so merge by index
df = df_features.copy()
df['reconstruction_mse'] = df_results['reconstruction_mse']
df['anomaly'] = df_results['anomaly']

# Extract full dataset name
df['dataset_name'] = df['dataset'].str.split('+').str[0]

# Extract classification label
df['classification'] = df['anomaly'].map({True: 'Anomaly', False: 'Normal'})

print("=" * 80)
print("TIME DOMAIN FEATURES BAR GRAPH ANALYSIS")
print("=" * 80)

FEATURES = ['rms', 'skewness', 'peak_to_peak', 'median', 'kurtosis', 'iqr', 'crest_factor', 'variance']

print(f"\nAnalyzing {len(FEATURES)} features across {df['dataset_name'].nunique()} datasets")
print(f"Total records: {len(df)}")
print(f"Normal: {(~df['anomaly']).sum()}, Anomaly: {df['anomaly'].sum()}")

# ============================================
# VISUALIZATION 1: Bar graphs by dataset
# ============================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, feature in enumerate(FEATURES):
    ax = axes[idx]

    # Calculate means for each dataset and classification
    means = df.groupby(['dataset_name', 'classification'])[feature].mean().unstack()

    # Plot grouped bar chart
    datasets = means.index
    x = np.arange(len(datasets))
    width = 0.35

    ax.bar(x - width/2, means['Normal'], width, label='Normal', color='#2ecc71', alpha=0.7)
    ax.bar(x + width/2, means['Anomaly'], width, label='Anomaly', color='#e74c3c', alpha=0.7)

    ax.set_title(f'{feature.upper()} Mean by Dataset & Classification', fontweight='bold')
    ax.set_xlabel('Dataset')
    ax.set_ylabel(f'Mean {feature}')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(title='Classification', loc='best')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('images/barplot_features_by_dataset.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: images/barplot_features_by_dataset.png")

# ============================================
# VISUALIZATION 2: Individual feature distributions
# ============================================

fig, axes = plt.subplots(1, 5, figsize=(18, 5))

for idx, feature in enumerate(FEATURES):
    ax = axes[idx]
    
    # Separate normal and anomaly
    normal_data = df[~df['anomaly']][feature]
    anomaly_data = df[df['anomaly']][feature]
    
    # Create violin plot
    parts = ax.violinplot([normal_data.dropna(), anomaly_data.dropna()], 
                          positions=[0, 1], showmeans=True, showmedians=True)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_ylabel(feature)
    ax.set_title(f'{feature.upper()}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('images/violinplot_features_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: images/violinplot_features_distribution.png")

# ============================================
# VISUALIZATION 3: Feature statistics by dataset and classification
# ============================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, feature in enumerate(FEATURES):
    ax = axes[idx]

    # Calculate statistics
    stats_data = []
    labels_list = []

    for dataset in sorted(df['dataset_name'].unique(t)):
        for classification in ['Normal', 'Anomaly']:
            subset = df[(df['dataset_name'] == dataset) & (df['classification'] == classification)]
            if len(subset) > 0:
                mean_val = subset[feature].mean()
                std_val = subset[feature].std()
                stats_data.append({'dataset': dataset, 'classification': classification,
                                 'mean': mean_val, 'std': std_val})
                labels_list.append(f'{dataset}\n{classification[0]}')

    stats_df = pd.DataFrame(stats_data)

    # Plot with error bars
    x_pos = np.arange(len(stats_df))
    colors = ['#2ecc71' if c == 'Normal' else '#e74c3c' for c in stats_df['classification']]

    ax.bar(x_pos, stats_df['mean'], yerr=stats_df['std'], capsize=5,
           color=colors, alpha=0.7, edgecolor='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_list, rotation=45, ha='right')
    ax.set_ylabel(f'Mean {feature}')
    ax.set_title(f'{feature.upper()} Mean ± Std', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('images/barplot_features_statistics.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: images/barplot_features_statistics.png")

# ============================================
# STATISTICAL SUMMARY TABLE
# ============================================

print("\n" + "=" * 80)
print("STATISTICAL SUMMARY BY DATASET AND CLASSIFICATION")
print("=" * 80)

for feature in FEATURES:
    print(f"\n{feature.upper()}")
    print("-" * 80)
    
    summary_stats = df.groupby(['dataset_name', 'classification'])[feature].agg([
        ('Count', 'count'),
        ('Mean', 'mean'),
        ('Std', 'std'),
        ('Min', 'min'),
        ('Q1', lambda x: x.quantile(0.25)),
        ('Median', 'median'),
        ('Q3', lambda x: x.quantile(0.75)),
        ('Max', 'max')
    ]).round(4)
    
    print(summary_stats.to_string())

# ============================================
# FEATURE COMPARISON: NORMAL vs ANOMALY
# ============================================

print("\n" + "=" * 80)
print("FEATURE COMPARISON: NORMAL vs ANOMALY (All Datasets Combined)")
print("=" * 80)

comparison_data = []
for feature in FEATURES:
    normal = df[~df['anomaly']][feature]
    anomaly = df[df['anomaly']][feature]
    
    comparison_data.append({
        'Feature': feature.upper(),
        'Normal_Mean': f"{normal.mean():.6f}",
        'Anomaly_Mean': f"{anomaly.mean():.6f}",
        'Normal_Std': f"{normal.std():.6f}",
        'Anomaly_Std': f"{anomaly.std():.6f}",
        'Difference_%': f"{((anomaly.mean() - normal.mean()) / normal.mean() * 100):.2f}%"
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# ============================================
# ANOMALY DETECTION EFFECTIVENESS
# ============================================

print("\n" + "=" * 80)
print("ANOMALY DETECTION INSIGHTS")
print("=" * 80)

for feature in FEATURES:
    normal = df[~df['anomaly']][feature].dropna()
    anomaly = df[df['anomaly']][feature].dropna()
    
    # Calculate separation
    normal_range = normal.max() - normal.min()
    anomaly_mean = anomaly.mean()
    normal_mean = normal.mean()
    
    print(f"\n{feature.upper()}:")
    print(f"  Normal range: [{normal.min():.4f}, {normal.max():.4f}]")
    print(f"  Anomaly mean: {anomaly_mean:.4f}")
    print(f"  Difference from normal mean: {(anomaly_mean - normal_mean):.4f} "
          f"({((anomaly_mean - normal_mean) / normal_mean * 100):+.2f}%)")
    
    # Check if anomalies have distinct pattern
    if anomaly_mean > normal.max():
        print(f"  → Anomalies have HIGHER values (clear separation)")
    elif anomaly_mean < normal.min():
        print(f"  → Anomalies have LOWER values (clear separation)")
    else:
        overlap = "Yes (some overlap with normal)"
        print(f"  → Anomalies overlap with normal range {overlap}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
