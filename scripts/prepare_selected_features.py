"""
Prepare a reduced CSV `selected_features.csv` with recommended features for autoencoder.
Columns: dataset,record_id,rms,peak_to_peak,skewness,kurtosis,qcd
Uses only Python stdlib so it can be run in minimal environments.
Run:
    python prepare_selected_features.py
"""
import csv
from pathlib import Path
import math

BASE = Path(__file__).resolve().parent.parent
IN = BASE / 'results' / 'extracted_features.csv'
OUT = BASE / 'results' / 'selected_features.csv'

cols_in = None
count = 0
with IN.open('r', encoding='utf-8') as fin, OUT.open('w', encoding='utf-8', newline='') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    header = next(reader)
    cols_in = header
    # map column name to index
    idx = {name: i for i, name in enumerate(cols_in)}
    wanted = ['dataset','record_id','rms','peak_to_peak','skewness','kurtosis','q25','q75']
    # write header (include qcd as final column)
    out_header = ['dataset','record_id','rms','peak_to_peak','skewness','kurtosis','qcd']
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
            q25 = get('q25')
            q75 = get('q75')
            denom = q75 + q25
            if denom == 0 or math.isnan(denom):
                qcd = float('nan')
            else:
                qcd = (q75 - q25) / denom
            # replace inf/nan with empty strings for CSV readability
            def safe(x):
                if x is None:
                    return ''
                if isinstance(x, float):
                    if math.isfinite(x):
                        return f"{x:.8g}"
                    else:
                        return ''
                return str(x)
            writer.writerow([dataset, record_id, safe(rms), safe(peak), safe(skew), safe(kurt), safe(qcd)])
            count += 1
        except Exception as e:
            # skip malformed rows
            continue

print(f"Wrote {count} rows to {OUT}")
