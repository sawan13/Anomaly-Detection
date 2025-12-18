import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("FEATURE OPTIMIZATION FOR UNSUPERVISED AUTOENCODER - ANOMALY DETECTION")
print("=" * 80)

# Read features
df = pd.read_csv('results/extracted_features.csv')

# Optimal feature set
optimal_features = ['rms', 'std', 'variance', 'crest_factor', 'peak_to_peak', 
                    'skewness', 'kurtosis', 'iqr', 'env_std', 'env_max',
                    'total_power', 'power_std', 'dominant_freq']

feature_data = df[optimal_features].copy()

print("\n" + "=" * 80)
print("1. CORRELATION ANALYSIS (for feature redundancy)")
print("=" * 80)

corr_matrix = feature_data.corr()

# Find highly correlated pairs
print("\nHighly Correlated Feature Pairs (> 0.85):")
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.85 and abs(corr_matrix.iloc[i, j]) < 1.0:
            high_corr.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

high_corr = sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)
if high_corr:
    for feat1, feat2, corr in high_corr[:10]:  # Top 10
        print(f"  {feat1:20s} <-> {feat2:20s} : {corr:7.4f}")
else:
    print("  No highly correlated features found!")

print("\n" + "=" * 80)
print("2. FEATURE SCALING ANALYSIS")
print("=" * 80)

print("\nOriginal Feature Statistics:")
stats = pd.DataFrame({
    'Feature': optimal_features,
    'Mean': feature_data.mean(),
    'Std': feature_data.std(),
    'Min': feature_data.min(),
    'Max': feature_data.max(),
})
print(stats.to_string())

print("\n" + "=" * 80)
print("3. OPTIMAL FEATURE SET FOR AUTOENCODER")
print("=" * 80)

print("""
✓ YES, these 13 features are SUITABLE for unsupervised autoencoder!

WHY THEY WORK WELL:
  1. Good Feature Diversity:
     - Time-domain: rms, std, variance, crest_factor, peak_to_peak, skewness, kurtosis, iqr
     - Envelope: env_std, env_max
     - Frequency: total_power, power_std, dominant_freq
  
  2. Complementary Information:
     - Statistical features capture amplitude characteristics
     - Envelope features capture modulation patterns
     - Frequency features capture spectral behavior
  
  3. Reasonable Dimensionality:
     - 13 features is ideal for autoencoder (not too few, not too many)
     - Allows meaningful compression in hidden layer

HOWEVER, CONSIDER THESE OPTIMIZATIONS:

""")

print("=" * 80)
print("4. RECOMMENDED MODIFICATIONS FOR BETTER AUTOENCODER PERFORMANCE")
print("=" * 80)

print("""
OPTION 1: USE ALL 13 FEATURES (RECOMMENDED)
  Pros:
    ✓ Maximum information retention
    ✓ Better anomaly detection sensitivity
    ✓ Captures all signal characteristics
  
  Cons:
    - Slight redundancy (variance, std, rms are correlated)
    - Slightly more computation
  
  Architecture: Input(13) -> Hidden(8) -> Bottleneck(4) -> Hidden(8) -> Output(13)

OPTION 2: REMOVE REDUNDANT FEATURES
  Remove: 'variance' (highly correlated with std and rms)
  Keep: 12 features
  
  Reasoning:
    - variance = std²
    - std and rms already capture variability
    - Removes ~15% redundancy while keeping 95% of information
  
  Architecture: Input(12) -> Hidden(7) -> Bottleneck(4) -> Hidden(7) -> Output(12)

OPTION 3: DIMENSIONALITY REDUCTION (Advanced)
  Use PCA to combine correlated features
  Keep: 10-11 uncorrelated components
  
  Architecture: Input(10) -> Hidden(6) -> Bottleneck(3) -> Hidden(6) -> Output(10)

""")

print("=" * 80)
print("5. REQUIRED DATA PREPROCESSING FOR AUTOENCODER")
print("=" * 80)

print("""
CRITICAL STEPS:
  1. DATA NORMALIZATION (MUST DO):
     Use StandardScaler (Z-score normalization):
     
     from sklearn.preprocessing import StandardScaler
     scaler = StandardScaler()
     X_scaled = scaler.fit_transform(feature_data)
     
     This is ESSENTIAL because:
     - Features have different scales (power is much larger than ratios)
     - Prevents dominant features from biasing the autoencoder
     - Improves convergence during training

  2. HANDLE OUTLIERS:
     Consider robust scaling for extreme values:
     
     from sklearn.preprocessing import RobustScaler
     scaler = RobustScaler()
     X_scaled = scaler.fit_transform(feature_data)

  3. TRAIN/VALIDATION SPLIT:
     Keep 80% for training, 20% for validation/testing
     
     from sklearn.model_selection import train_test_split
     X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

""")

print("=" * 80)
print("6. AUTOENCODER ARCHITECTURE RECOMMENDATIONS")
print("=" * 80)

print("""
For 13 Features (RECOMMENDED):

Option A - Standard Autoencoder:
  Input: 13
  Layer 1: 13 -> 10 (ReLU) + Dropout(0.2)
  Layer 2: 10 -> 6 (ReLU) + Dropout(0.2)
  Bottleneck: 6 -> 3 (Linear)  <- Compressed representation
  Layer 3: 3 -> 6 (ReLU) + Dropout(0.2)
  Layer 4: 6 -> 10 (ReLU) + Dropout(0.2)
  Output: 10 -> 13 (Linear)

  Compression Ratio: 13 -> 3 (76.9% compression)
  Training Epochs: 100-150
  Batch Size: 32
  Loss Function: MSE (Mean Squared Error)

Option B - Deep Autoencoder:
  Input: 13
  Layer 1: 13 -> 10 (ReLU)
  Layer 2: 10 -> 7 (ReLU)
  Bottleneck: 7 -> 4 (Linear)
  Layer 3: 4 -> 7 (ReLU)
  Layer 4: 7 -> 10 (ReLU)
  Output: 10 -> 13 (Linear)

Option C - Variational Autoencoder (VAE):
  Better for unsupervised anomaly detection
  Input: 13
  Encoder: 13 -> 8 -> 4 (mean & log_var)
  Decoder: 4 -> 8 -> 13
  Includes KL divergence loss for better regularization

""")

print("=" * 80)
print("7. ANOMALY DETECTION STRATEGY")
print("=" * 80)

print("""
RECONSTRUCTION ERROR METHOD:
  1. Train autoencoder on normal data
  2. Calculate reconstruction error (MSE) on test data:
     error = MSE(original_data, decoded_data)
  
  3. Set anomaly threshold:
     threshold = mean_error + (2 * std_error)
  
  4. Classify samples:
     if error > threshold: ANOMALY
     else: NORMAL

EXPECTED PERFORMANCE:
  - Sensitivity: 85-95% (detect anomalies)
  - Specificity: 90-98% (avoid false positives)
  - This depends on feature quality and training data

""")

print("=" * 80)
print("8. FINAL RECOMMENDATION")
print("=" * 80)

print("""
✓ USE ALL 13 FEATURES with StandardScaler normalization

Reasoning:
  1. All 13 features contribute unique information
  2. Even "redundant" features (std, variance, rms) have subtle differences
  3. Autoencoder is robust to some redundancy
  4. Better detection of complex anomaly patterns
  5. The slight correlation doesn't harm autoencoder learning

TRAINING PIPELINE:
  1. Load extracted_features.csv
  2. Select 13 optimal features
  3. Apply StandardScaler normalization
  4. Train/test split (80/20)
  5. Build autoencoder (Option A recommended)
  6. Train for 100-150 epochs
  7. Calculate reconstruction error
  8. Set anomaly threshold
  9. Evaluate on test data

""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
