Usage: train an autoencoder on `selected_features.csv`.

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Train the autoencoder:

```bash
python train_autoencoder_selected.py
```

Outputs:
- `reconstruction_selected.csv` with reconstruction MSE and anomaly flag
- `autoencoder_selected.joblib` saved scaler+model
- `reconstruction_selected_hist.png` (if matplotlib installed)

Notes:
- The script uses a simple MLP autoencoder. Adjust architecture or preprocessing as needed.
