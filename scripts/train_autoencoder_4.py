"""
Train autoencoder on 4 selected features for anomaly detection.
Features: ['rms','peak_to_peak','skewness','kurtosis']

Usage:
    python train_autoencoder_4.py

Notes:
- Requires: pandas, numpy, scikit-learn, matplotlib
- Saves reconstruction results to reconstruction_4.csv
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

FEATURES = ['rms','peak_to_peak','skewness','kurtosis']
INPUT_CSV = 'results/selected_features_4.csv'
OUTPUT_CSV = 'results/reconstruction_4.csv'
PLOT_PNG = 'results/reconstruction_error_hist_4.png'

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

    # Split data: 70% train, 15% val, 15% test
    X_train, X_temp, indices_train, indices_temp = train_test_split(X, data.index, test_size=0.3, random_state=42)
    X_val, X_test, indices_val, indices_test = train_test_split(X_temp, indices_temp, test_size=0.5, random_state=42)

    # Build and train autoencoder
    input_dim = X.shape[1]
    model = build_autoencoder(input_dim=input_dim, encoding_dim=3)

    print("Training MLP autoencoder on 4 selected features...")
    model.fit(X_train, X_train)

    # Compute losses
    X_train_pred = model.predict(X_train)
    mse_train = np.mean(np.power(X_train - X_train_pred, 2), axis=1)
    train_loss = mse_train.mean()

    X_val_pred = model.predict(X_val)
    mse_val = np.mean(np.power(X_val - X_val_pred, 2), axis=1)
    val_loss = mse_val.mean()

    X_test_pred = model.predict(X_test)
    mse_test = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
    test_loss = mse_test.mean()

    # Final losses
    final_train_loss = model.loss_
    final_val_loss = model.best_loss_

    # Epochs
    epochs = model.n_iter_

    # Model parameters
    model_params = sum(p.size for p in model.coefs_) + sum(p.size for p in model.intercepts_)

    # Ratios
    train_val_ratio = final_train_loss / final_val_loss if final_val_loss != 0 else 0
    test_train_ratio = test_loss / train_loss if train_loss != 0 else 0
    val_train_ratio = val_loss / train_loss if train_loss != 0 else 0

    # Std devs
    train_std = mse_train.std()
    val_std = mse_val.std()
    test_std = mse_test.std()

    # Print summary
    print(f"\n{'='*60}")
    print("4-Feature Model Autoencoder Analysis")
    print(f"{'='*60}")
    print(f"Model Parameters: {model_params}")
    print(f"Epochs Trained: {epochs}")
    print(f"Final Training Loss: {final_train_loss:.6f}")
    print(f"Final Validation Loss: {final_val_loss:.6f}")
    print(f"Train/Val Loss Ratio: {train_val_ratio:.4f}")
    print(f"Mean Train Error (MSE): {train_loss:.6f}")
    print(f"Mean Val Error (MSE): {val_loss:.6f}")
    print(f"Mean Test Error (MSE): {test_loss:.6f}")
    print(f"Test/Train Error Ratio: {test_train_ratio:.4f}")
    print(f"Val/Train Error Ratio: {val_train_ratio:.4f}")
    print(f"Train Error Std Dev: {train_std:.6f}")
    print(f"Val Error Std Dev: {val_std:.6f}")
    print(f"Test Error Std Dev: {test_std:.6f}")

    # Now, for reconstruction on full data
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
    print(f"\nTotal records processed: {total}")
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
