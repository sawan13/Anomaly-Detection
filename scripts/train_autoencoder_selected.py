"""
Train autoencoder on selected features for anomaly detection.
Features: ['rms','peak_to_peak','skewness','kurtosis','qcd']

Usage:
    python train_autoencoder_selected.py

Notes:
- Requires: pandas, numpy, scikit-learn, matplotlib
- Saves reconstruction results to reconstruction_selected.csv
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

FEATURES = ['rms','peak_to_peak','skewness','kurtosis','qcd']
INPUT_CSV = 'results/selected_features.csv'
OUTPUT_CSV = 'results/reconstruction_selected.csv'
PLOT_PNG = 'results/reconstruction_error_hist_selected.png'

# Check dependencies
try:
    from sklearn.neural_network import MLPRegressor
except Exception:
    print("Missing scikit-learn. Install it in your venv or conda env:")
    print("  pip install scikit-learn")
    print("  or\n  conda install -c conda-forge scikit-learn")
    raise


def build_autoencoder(input_dim, encoding_dim=3):
    """Return an MLPRegressor configured as an autoencoder."""
    hidden1 = max(input_dim, 8)
    hidden2 = encoding_dim
    hidden3 = hidden1
    model = MLPRegressor(
        hidden_layer_sizes=(hidden1, hidden2, hidden3),
        activation='relu',
        solver='adam',
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    )
    return model


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Input file not found: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    # Verify features exist
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print("Missing features in CSV:", missing)
        sys.exit(1)

    data = df[FEATURES].copy()
    data = data.dropna()

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(data.values)

    # Build and train autoencoder
    input_dim = X.shape[1]
    model = build_autoencoder(input_dim=input_dim, encoding_dim=3)

    print("Training MLP autoencoder on selected features...")
    model.fit(X, X)

    # Reconstruct and compute MSE
    X_pred = model.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=1)

    # Threshold: mean + 2*std
    thresh = mse.mean() + 2 * mse.std()

    # Map results back
    data_idx = data.index

    results = pd.DataFrame({
        'index': data_idx,
        'reconstruction_mse': mse
    })
    results['is_anomaly'] = results['reconstruction_mse'] > thresh

    # Save results
    out_df = df.loc[results['index']].copy()
    out_df = out_df.reset_index(drop=True)
    results = results.reset_index(drop=True)
    merged = pd.concat([out_df, results[['reconstruction_mse','is_anomaly']]], axis=1)
    merged.to_csv(OUTPUT_CSV, index=False)

    # Print summary
    n_anom = merged['is_anomaly'].sum()
    total = len(merged)
    print(f"\n{'='*60}")
    print("Autoencoder Reconstruction Analysis (Selected Features)")
    print(f"{'='*60}")
    print(f"Total records processed: {total}")
    print(f"Anomalies detected (threshold={thresh:.6e}): {n_anom} ({n_anom/total:.2%})")
    print(f"Normal records: {total - n_anom} ({(total-n_anom)/total:.2%})")
    print(f"\nReconstruction Error Statistics:")
    print(f"  Mean: {mse.mean():.6e}")
    print(f"  Std:  {mse.std():.6e}")
    print(f"  Min:  {mse.min():.6e}")
    print(f"  Max:  {mse.max():.6e}")
    print(f"\nSaved reconstruction results to: {OUTPUT_CSV}")

    # Plot histogram
    plt.figure(figsize=(8,5))
    plt.hist(mse, bins=50, alpha=0.7, label='MSE')
    plt.axvline(thresh, color='r', linestyle='--', label=f'threshold={thresh:.2e}')
    plt.xlabel('Reconstruction MSE')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PNG)
    print(f"Saved reconstruction error histogram to: {PLOT_PNG}")


if __name__ == '__main__':
    main()
