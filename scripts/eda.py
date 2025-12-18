import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile
import os
import warnings
from scipy import signal
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Extract ZIP files
zip_path = r'd:\Anomaly dataset'
extract_path = r'd:\Anomaly dataset\extracted'

# Create extract directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract all ZIP files
zip_files = ['BSK.362+GB01-N01.zip', 'BIS.361+GB01-N01.zip', 'AGR.561+GB01-N01.zip', 'AGR.361+GB01-N01.zip']
datasets = {}

for zip_file in zip_files:
    zip_file_path = os.path.join(zip_path, zip_file)
    if os.path.exists(zip_file_path):
        with ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            print(f"✓ Extracted {zip_file}")

# Load all datasets
import json

for folder in os.listdir(extract_path):
    folder_path = os.path.join(extract_path, folder)
    if os.path.isdir(folder_path):
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        if json_files:
            # Load and combine all JSON files from this folder
            data_list = []
            for json_file in json_files:
                json_path = os.path.join(folder_path, json_file)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            data_list.extend(data)
                        else:
                            data_list.append(data)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
            
            if data_list:
                datasets[folder] = pd.DataFrame(data_list)
                print(f"✓ Loaded {folder}: Shape {datasets[folder].shape}")

# EDA for each dataset
for name, data in datasets.items():
    print(f"\n{'='*60}")
    print(f"EDA for {name}")
    print(f"{'='*60}")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"\nData types:\n{data.dtypes}")
    print(f"\nMissing values:\n{data.isnull().sum()}")

# ============================================
# FEATURE EXTRACTION from nested data
# ============================================

print(f"\n\n{'='*60}")
print("FEATURE EXTRACTION FROM SENSOR DATA")
print(f"{'='*60}")

comparison_data = []

for dataset_name, data in datasets.items():
    print(f"\nProcessing {dataset_name}...")
    
    # Extract short name (e.g., "AGR.361" from "AGR.361+GB01-N01")
    short_name = dataset_name.split('+')[0]
    
    for idx, row in data.iterrows():
        try:
            # Extract sensor data array
            sensor_data = row['data']
            if isinstance(sensor_data, list):
                sensor_array = np.array(sensor_data, dtype=float)
                
                # ============================================
                # TIME DOMAIN FEATURES
                # ============================================
                
                # Basic statistics
                mean_val = np.mean(sensor_array)
                std_val = np.std(sensor_array)
                min_val = np.min(sensor_array)
                max_val = np.max(sensor_array)
                median_val = np.median(sensor_array)
                
                # RMS (Root Mean Square)
                rms = np.sqrt(np.mean(sensor_array**2))
                
                # Variance
                variance = np.var(sensor_array)
                
                # Peak-to-peak
                peak_to_peak = max_val - min_val
                
                # Crest Factor (Peak / RMS)
                crest_factor = max_val / rms if rms != 0 else 0
                
                # Skewness and Kurtosis
                skewness = skew(sensor_array)
                kurt = kurtosis(sensor_array)
                
                # Quartiles
                q25 = np.percentile(sensor_array, 25)
                q75 = np.percentile(sensor_array, 75)
                iqr = q75 - q25
                
                # ============================================
                # ENVELOPE FEATURES
                # ============================================
                
                # Calculate envelope using Hilbert transform
                analytic_signal = signal.hilbert(sensor_array)
                envelope = np.abs(analytic_signal)
                env_mean = np.mean(envelope)
                env_std = np.std(envelope)
                env_max = np.max(envelope)
                env_min = np.min(envelope)
                
                # ============================================
                # FREQUENCY DOMAIN FEATURES
                # ============================================
                
                # FFT
                fft_vals = np.fft.fft(sensor_array)
                magnitude = np.abs(fft_vals)
                power = magnitude**2
                
                # Power spectrum features
                total_power = np.sum(power)
                power_mean = np.mean(power)
                power_std = np.std(power)
                
                # Dominant frequency
                dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
                dominant_freq = dominant_freq_idx / len(sensor_array)
                
                # Extract features from sensor data
                features = {
                    'dataset': short_name,
                    'record_id': idx,
                    
                    # TIME DOMAIN FEATURES
                    'mean': mean_val,
                    'std': std_val,
                    'variance': variance,
                    'min': min_val,
                    'max': max_val,
                    'median': median_val,
                    'rms': rms,
                    'peak_to_peak': peak_to_peak,
                    'crest_factor': crest_factor,
                    'skewness': skewness,
                    'kurtosis': kurt,
                    'q25': q25,
                    'q75': q75,
                    'iqr': iqr,
                    
                    # ENVELOPE FEATURES
                    'env_mean': env_mean,
                    'env_std': env_std,
                    'env_max': env_max,
                    'env_min': env_min,
                    
                    # FREQUENCY DOMAIN FEATURES
                    'total_power': total_power,
                    'power_mean': power_mean,
                    'power_std': power_std,
                    'dominant_freq': dominant_freq,
                    
                    # ORIGINAL METADATA
                    'data_length': len(sensor_array),
                    'shaftSpeed': row['shaftSpeed'],
                    'samplingRate': row['samplingRate'],
                }
                comparison_data.append(features)
        except Exception as e:
            print(f"  Error processing row {idx}: {e}")

# Create comparison dataframe
comparison_df = pd.DataFrame(comparison_data)
print(f"\nExtracted {len(comparison_df)} records with {len(comparison_df.columns)} features")
print(f"\nFeatures extracted:")
print(f"  - Time Domain: mean, std, variance, rms, skewness, kurtosis, crest_factor, peak_to_peak, iqr")
print(f"  - Envelope: env_mean, env_std, env_max, env_min")
print(f"  - Frequency Domain: total_power, power_mean, power_std, dominant_freq")
print(f"\nFeature Statistics:\n{comparison_df.describe()}")

# ============================================
# VISUALIZATIONS FOR DATASET COMPARISON
# ============================================

# FIGURE 1: Basic Statistics
fig1 = plt.figure(figsize=(15, 5))
gs1 = fig1.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

# 1. Record count by dataset
ax1 = fig1.add_subplot(gs1[0, 0])
dataset_counts = comparison_df['dataset'].value_counts()
dataset_counts.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Record Count by Dataset', fontsize=12, fontweight='bold')
ax1.set_xlabel('Dataset')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# 2. Mean value comparison
ax2 = fig1.add_subplot(gs1[0, 1])
sns.boxplot(data=comparison_df, x='dataset', y='mean', ax=ax2, palette='Set2')
ax2.set_title('Mean Sensor Values by Dataset', fontsize=12, fontweight='bold')
ax2.set_xlabel('Dataset')
ax2.set_ylabel('Mean')
ax2.tick_params(axis='x', rotation=45)

# 3. Standard deviation comparison
ax3 = fig1.add_subplot(gs1[0, 2])
sns.boxplot(data=comparison_df, x='dataset', y='std', ax=ax3, palette='Set1')
ax3.set_title('Std Dev by Dataset', fontsize=12, fontweight='bold')
ax3.set_xlabel('Dataset')
ax3.set_ylabel('Std Dev')
ax3.tick_params(axis='x', rotation=45)

fig1.suptitle('Figure 1: Dataset Count & Basic Statistics', fontsize=14, fontweight='bold', y=1.02)
plt.show()

# FIGURE 2: Distribution Analysis
fig2 = plt.figure(figsize=(15, 5))
gs2 = fig2.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

# 4. Min-Max range
ax4 = fig2.add_subplot(gs2[0, 0])
comparison_df['range'] = comparison_df['max'] - comparison_df['min']
sns.boxplot(data=comparison_df, x='dataset', y='range', ax=ax4, palette='husl')
ax4.set_title('Value Range (Max - Min) by Dataset', fontsize=12, fontweight='bold')
ax4.set_xlabel('Dataset')
ax4.set_ylabel('Range')
ax4.tick_params(axis='x', rotation=45)

# 5. Data length distribution
ax5 = fig2.add_subplot(gs2[0, 1])
sns.violinplot(data=comparison_df, x='dataset', y='data_length', ax=ax5, palette='muted')
ax5.set_title('Sensor Data Length Distribution', fontsize=12, fontweight='bold')
ax5.set_xlabel('Dataset')
ax5.set_ylabel('Data Length')
ax5.tick_params(axis='x', rotation=45)

# 6. Mean vs Std scatter plot
ax6 = fig2.add_subplot(gs2[0, 2])
colors = {'AGR.361': '#FF6B6B', 'AGR.561': '#4ECDC4', 
          'BIS.361': '#45B7D1', 'BSK.362': '#FFA07A'}
for dataset in comparison_df['dataset'].unique():
    data_subset = comparison_df[comparison_df['dataset'] == dataset]
    ax6.scatter(data_subset['mean'], data_subset['std'], label=dataset, s=80, alpha=0.7, 
               color=colors.get(dataset, 'gray'))
ax6.set_title('Mean vs Std Deviation', fontsize=12, fontweight='bold')
ax6.set_xlabel('Mean')
ax6.set_ylabel('Std Dev')
ax6.legend(fontsize=9, loc='best')
ax6.grid(True, alpha=0.3)

fig2.suptitle('Figure 2: Distribution & Relationship Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.show()

# FIGURE 3: Advanced Comparisons
fig3 = plt.figure(figsize=(15, 5))
gs3 = fig3.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

# 7. Quartile comparison
ax7 = fig3.add_subplot(gs3[0, 0])
quartile_data = comparison_df.groupby('dataset')[['q25', 'median', 'q75']].mean()
quartile_data.plot(kind='bar', ax=ax7, width=0.8, color=['#FF9999', '#66B2FF', '#99FF99'])
ax7.set_title('Average Quartiles by Dataset', fontsize=12, fontweight='bold')
ax7.set_xlabel('Dataset')
ax7.set_ylabel('Value')
ax7.tick_params(axis='x', rotation=45)
ax7.legend(['Q25', 'Median', 'Q75'], fontsize=10)
ax7.grid(True, alpha=0.3, axis='y')

# 8. Distribution of means
ax8 = fig3.add_subplot(gs3[0, 1])
for dataset in comparison_df['dataset'].unique():
    data_subset = comparison_df[comparison_df['dataset'] == dataset]
    ax8.hist(data_subset['mean'], alpha=0.6, label=dataset, bins=15)
ax8.set_title('Distribution of Mean Values', fontsize=12, fontweight='bold')
ax8.set_xlabel('Mean')
ax8.set_ylabel('Frequency')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3, axis='y')

# 9. Dataset statistics table
ax9 = fig3.add_subplot(gs3[0, 2])
ax9.axis('off')
stats_summary = []
for dataset in sorted(comparison_df['dataset'].unique()):
    data_subset = comparison_df[comparison_df['dataset'] == dataset]
    stats_summary.append([
        dataset,
        f"{data_subset['mean'].mean():.2f}",
        f"{data_subset['std'].mean():.2f}",
        f"{data_subset['data_length'].mean():.0f}",
        f"{len(data_subset)}"
    ])

table = ax9.table(cellText=stats_summary, 
                  colLabels=['Dataset', 'Avg Mean', 'Avg Std', 'Avg Length', 'Records'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(stats_summary) + 1):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('#ffffff')

ax9.set_title('Statistical Summary', fontsize=12, fontweight='bold', pad=20)

fig3.suptitle('Figure 3: Quartiles, Distributions & Summary', fontsize=14, fontweight='bold', y=1.02)
plt.show()

print("\n✓ All 3 visualizations complete!")

# ============================================
# SAVE FEATURES TO CSV
# ============================================

output_path = r'd:\Anomaly dataset\extracted_features.csv'

try:
    comparison_df.to_csv(output_path, index=False)
    print(f"\n✓ Features saved to: {output_path}")
except PermissionError:
    print(f"\n⚠ File is locked (possibly open in Excel)")
    # Try alternate filename
    output_path = r'd:\Anomaly dataset\extracted_features_new.csv'
    comparison_df.to_csv(output_path, index=False)
    print(f"✓ Features saved to: {output_path}")
    print("  (Close the original file in Excel and rename if needed)")

print(f"\nDataFrame shape: {comparison_df.shape}")
print(f"Columns: {list(comparison_df.columns)}")