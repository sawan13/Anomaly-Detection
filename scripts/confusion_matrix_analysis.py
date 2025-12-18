"""
Anomaly Detection Results Analysis by Dataset
Shows confusion matrix and normal vs anomaly breakdown by AGR, BIS, BSK
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = r'd:\Anomaly dataset\reconstruction_results.csv'

# Load results
df = pd.read_csv(INPUT_CSV)

print("=" * 80)
print("ANOMALY DETECTION RESULTS ANALYSIS")
print("=" * 80)

# Extract dataset name (AGR.361, AGR.561, BIS.361, BSK.362)
# Use full dataset name for detailed breakdown
df['dataset_name'] = df['dataset'].str.split('+').str[0]

print(f"\nTotal records analyzed: {len(df)}")
print(f"Total anomalies detected: {df['anomaly'].sum()}")
print(f"Total normal records: {(~df['anomaly']).sum()}")
print(f"Anomaly rate: {df['anomaly'].sum()/len(df):.2%}")

# ============================================
# BREAKDOWN BY DATASET
# ============================================

print("\n" + "=" * 80)
print("BREAKDOWN BY DATASET")
print("=" * 80)

dataset_stats = df.groupby('dataset_name').agg({
    'anomaly': ['sum', 'count'],
    'reconstruction_mse': ['mean', 'std', 'max']
}).round(6)

# Flatten multi-level columns
dataset_stats.columns = ['anomalies', 'total_records', 'mean_mse', 'std_mse', 'max_mse']
dataset_stats['normal_records'] = dataset_stats['total_records'] - dataset_stats['anomalies']
dataset_stats['anomaly_rate'] = (dataset_stats['anomalies'] / dataset_stats['total_records'] * 100).round(2)
dataset_stats['normal_rate'] = (dataset_stats['normal_records'] / dataset_stats['total_records'] * 100).round(2)

# Reorder columns for readability
dataset_stats = dataset_stats[['total_records', 'normal_records', 'anomalies', 
                               'normal_rate', 'anomaly_rate', 'mean_mse', 'std_mse', 'max_mse']]

print("\n" + dataset_stats.to_string())

# ============================================
# CONFUSION MATRIX STYLE TABLE
# ============================================

print("\n" + "=" * 80)
print("CLASSIFICATION SUMMARY (NORMAL vs ANOMALY)")
print("=" * 80)

confusion = pd.DataFrame({
    'Dataset': dataset_stats.index,
    'Normal': dataset_stats['normal_records'].astype(int),
    'Anomaly': dataset_stats['anomalies'].astype(int),
    'Total': dataset_stats['total_records'].astype(int)
})

# Add totals row
totals_row = pd.DataFrame({
    'Dataset': ['TOTAL'],
    'Normal': [dataset_stats['normal_records'].sum()],
    'Anomaly': [dataset_stats['anomalies'].sum()],
    'Total': [dataset_stats['total_records'].sum()]
})

confusion = pd.concat([confusion, totals_row], ignore_index=True)
print("\n" + confusion.to_string(index=False))

# ============================================
# VISUALIZATIONS
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Stacked bar chart (Normal vs Anomaly by dataset)
ax1 = axes[0, 0]
dataset_names = dataset_stats.index.tolist()
normal_counts = dataset_stats['normal_records'].values
anomaly_counts = dataset_stats['anomalies'].values

x = np.arange(len(dataset_names))
width = 0.6

ax1.bar(x, normal_counts, width, label='Normal', color='#2ecc71', alpha=0.8)
ax1.bar(x, anomaly_counts, width, bottom=normal_counts, label='Anomaly', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Dataset')
ax1.set_ylabel('Record Count')
ax1.set_title('Normal vs Anomaly Distribution by Dataset', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(dataset_names)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (n, a) in enumerate(zip(normal_counts, anomaly_counts)):
    ax1.text(i, n/2, str(int(n)), ha='center', va='center', fontweight='bold', color='white')
    ax1.text(i, n + a/2, str(int(a)), ha='center', va='center', fontweight='bold', color='white')

# 2. Percentage bar chart
ax2 = axes[0, 1]
normal_pct = (normal_counts / (normal_counts + anomaly_counts) * 100)
anomaly_pct = (anomaly_counts / (normal_counts + anomaly_counts) * 100)

ax2.bar(x, normal_pct, width, label='Normal', color='#2ecc71', alpha=0.8)
ax2.bar(x, anomaly_pct, width, bottom=normal_pct, label='Anomaly', color='#e74c3c', alpha=0.8)

ax2.set_xlabel('Dataset')
ax2.set_ylabel('Percentage (%)')
ax2.set_title('Anomaly Rate (%) by Dataset', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(dataset_names)
ax2.legend()
ax2.set_ylim([0, 105])
ax2.grid(True, alpha=0.3, axis='y')

# Add percentage labels
for i, (n_pct, a_pct) in enumerate(zip(normal_pct, anomaly_pct)):
    ax2.text(i, n_pct/2, f'{n_pct:.1f}%', ha='center', va='center', fontweight='bold', color='white')
    if a_pct > 3:  # Only show if visible
        ax2.text(i, n_pct + a_pct/2, f'{a_pct:.1f}%', ha='center', va='center', fontweight='bold', color='white')

# 3. Box plot of reconstruction MSE by dataset and classification
ax3 = axes[1, 0]
mse_data = []
labels = []
colors = []
for dataset_name in sorted(dataset_stats.index):
    subset = df[df['dataset_name'] == dataset_name]
    normal_mse = subset[~subset['anomaly']]['reconstruction_mse'].values
    anomaly_mse = subset[subset['anomaly']]['reconstruction_mse'].values
    
    if len(normal_mse) > 0:
        mse_data.append(normal_mse)
        labels.append(f'{dataset_name}\n(Normal)')
        colors.append('#2ecc71')
    
    if len(anomaly_mse) > 0:
        mse_data.append(anomaly_mse)
        labels.append(f'{dataset_name}\n(Anomaly)')
        colors.append('#e74c3c')

bp = ax3.boxplot(mse_data, labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.set_ylabel('Reconstruction MSE')
ax3.set_title('Reconstruction Error Distribution (by Full Dataset)', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.tick_params(axis='x', rotation=45)

# 4. Confusion matrix heatmap style
ax4 = axes[1, 1]

confusion_matrix = dataset_stats[['normal_records', 'anomalies']].astype(int)
confusion_matrix.columns = ['Normal', 'Anomaly']

sns.heatmap(confusion_matrix.T, annot=True, fmt='d', cmap='RdYlGn_r', 
            cbar_kws={'label': 'Count'}, ax=ax4, xticklabels=confusion_matrix.index,
            linewidths=0.5, linecolor='gray')

ax4.set_title('Confusion Matrix - Count by Full Dataset Name', fontweight='bold')
ax4.set_ylabel('Classification')
ax4.set_xlabel('Dataset (Full Name)')

plt.tight_layout()
plt.savefig(r'd:\Anomaly dataset\anomaly_analysis_by_dataset.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization to: d:\\Anomaly dataset\\anomaly_analysis_by_dataset.png")

# ============================================
# DETAILED ANOMALY EXAMPLES
# ============================================

print("\n" + "=" * 80)
print("SAMPLE ANOMALIES DETECTED")
print("=" * 80)

for dataset_name in sorted(dataset_stats.index):
    anomalies = df[(df['dataset_name'] == dataset_name) & (df['anomaly'])]
    
    if len(anomalies) > 0:
        print(f"\n{dataset_name}: {len(anomalies)} anomalies found")
        print("-" * 60)
        
        # Show top 3 anomalies by highest MSE
        top_anomalies = anomalies.nlargest(3, 'reconstruction_mse')[
            ['dataset', 'record_id', 'rms', 'std', 'reconstruction_mse']
        ]
        
        for idx, row in top_anomalies.iterrows():
            print(f"  Record {row['record_id']:3d} | MSE: {row['reconstruction_mse']:.6e} | "
                  f"RMS: {row['rms']:.4f} | STD: {row['std']:.4f}")
    else:
        print(f"\n{dataset_name}: No anomalies detected")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
