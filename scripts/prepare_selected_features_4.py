"""
Prepare a reduced CSV `selected_features_4.csv` with 4 recommended features for autoencoder.
Columns: dataset,record_id,rms,peak_to_peak,skewness,kurtosis
Uses only Python stdlib so it can be run in minimal environments.
Run:
    python prepare_selected_features_4.py
"""
import csv
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
IN = BASE / 'results' / 'extracted_features.csv'
OUT = BASE / 'results' / 'selected_features_4.csv'

def safe(x):
    if x is None:
        return ''
    if isinstance(x, float):
        if math.isfinite(x):
            return f"{x:.8g}"
        else:
            return ''
    return str(x)

cols_in = None
count = 0
with IN.open('r', encoding='utf-8') as fin, OUT.open('w', encoding='utf-8', newline='') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    header = next(reader)
    cols_in = header
    # map column name to index
    idx = {name: i for i, name in enumerate(cols_in)}
    wanted = ['dataset','record_id','rms','peak_to_peak','skewness','kurtosis']
    # write header
    out_header = ['dataset','record_id','rms','peak_to_peak','skewness','kurtosis']
    writer.writerow(out_header)

    for row in reader:
        try:
            dataset = row[idx['dataset']]
            record_id = row[idx['record_id']]
            def get(col):
                if col in idx:
                    v = row[idx[col]].strip()
                    return float(v) if v not in ('', 'nan') else float('nan')
                return float('nan')
            rms = get('rms')
            peak = get('peak_to_peak')
            skew = get('skewness')
            kurt = get('kurtosis')
            writer.writerow([dataset, record_id, safe(rms), safe(peak), safe(skew), safe(kurt)])
            count += 1
        except Exception as e:
            # skip malformed rows
            continue

print(f"Wrote {count} rows to {OUT}")
