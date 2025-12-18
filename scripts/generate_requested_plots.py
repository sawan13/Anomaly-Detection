"""
Generate requested feature bar plots and correlation heatmap.
Requires: pandas, matplotlib, seaborn
Run:
  - Activate your Python environment (conda/venv)
  - Install dependencies: `pip install pandas matplotlib seaborn` or use conda
  - Run: `python generate_requested_plots.py`
Output files:
  - images/requested_features_barplot.png
  - images/feature_correlation.png
  - results/requested_features_summary_by_dataset.csv
  - results/feature_correlation.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(__file__)
CSV_IN = os.path.join(BASE, '..', 'results', 'extracted_features.csv')
OUT_BAR_PNG = os.path.join(BASE, '..', 'images', 'requested_features_barplot.png')
OUT_CORR_PNG = os.path.join(BASE, '..', 'images', 'feature_correlation.png')
OUT_SUMMARY = os.path.join(BASE, '..', 'results', 'requested_features_summary_by_dataset.csv')
OUT_CORR_CSV = os.path.join(BASE, '..', 'results', 'feature_correlation.csv')

requested = ['rms', 'std', 'skewness', 'peak_to_peak', 'qcd', 'kurtosis', 'iqr', 'variance', 'crest_factor']
previous = ['rms', 'std', 'variance', 'crest_factor', 'peak_to_peak']
all_features = list(dict.fromkeys(requested + previous))

# Load
print('Loading', CSV_IN)
df = pd.read_csv(CSV_IN)

# Compute QCD if q25/q75 available
if 'q25' in df.columns and 'q75' in df.columns:
    denom = (df['q75'] + df['q25']).replace(0, np.nan)
    df['qcd'] = (df['q75'] - df['q25']) / denom
    df['qcd'].replace([np.inf, -np.inf], np.nan, inplace=True)
else:
    df['qcd'] = np.nan

# Ensure columns exist
for c in all_features:
    if c not in df.columns:
        df[c] = np.nan

# Summary by dataset
summary = df.groupby('dataset')[requested].agg(['mean','std'])
summary.columns = ['_'.join(col) for col in summary.columns]
summary.reset_index(inplace=True)
summary.to_csv(OUT_SUMMARY, index=False)
print('Saved summary:', OUT_SUMMARY)

# Bar plots: try to use seaborn style, fall back gracefully if unavailable
style_name = 'seaborn-whitegrid'
try:
    if style_name in plt.style.available:
        plt.style.use(style_name)
    else:
        try:
            # prefer seaborn theme if seaborn is installed
            sns.set_theme(style='whitegrid')
        except Exception:
            plt.style.use('default')
            print(f"Warning: '{style_name}' style not available; using default style.")
except Exception as e:
    print(f"Warning: unable to set plotting style ({e}); using default.")
    plt.style.use('default')
fig, axes = plt.subplots(1, len(requested), figsize=(4*len(requested),5), constrained_layout=True)
if len(requested) == 1:
    axes = [axes]

datasets = summary['dataset'].tolist()
ind = np.arange(len(datasets))

for i, feat in enumerate(requested):
    means = summary[f'{feat}_mean'].values
    errs = summary[f'{feat}_std'].values
    ax = axes[i]
    colors = sns.color_palette('tab10')
    ax.bar(ind, means, yerr=errs, capsize=5, color=colors[:len(datasets)])
    ax.set_xticks(ind)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_title(feat)
    ax.set_ylabel('Value')

plt.suptitle('Requested Feature Means by Dataset (error bars = std)')
plt.savefig(OUT_BAR_PNG, dpi=400)
plt.close(fig)
print('Saved plot:', OUT_BAR_PNG)

# Correlation matrix
corr = df[all_features].corr()
corr.to_csv(OUT_CORR_CSV)
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(OUT_CORR_PNG, dpi=400)
plt.close()
print('Saved correlation:', OUT_CORR_PNG, OUT_CORR_CSV)

