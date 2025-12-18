"""
Analyze `reconstruction_selected.csv` produced by `train_autoencoder_selected.py`.
Generates:
 - `anomaly_summary.txt` : overall counts and per-dataset counts
 - `top_anomalies.csv` : top N samples by reconstruction_mse
 - `anomaly_counts_by_dataset.png` (if matplotlib available)

Run:
  python analyze_reconstruction_selected.py
"""
from pathlib import Path
import csv
import sys

BASE = Path(__file__).resolve().parent
IN_CSV = BASE / 'reconstruction_selected.csv'
OUT_SUM = BASE / 'anomaly_summary.txt'
OUT_TOP = BASE / 'top_anomalies.csv'
OUT_PLOT = BASE / 'anomaly_counts_by_dataset.png'

if not IN_CSV.exists():
    print('Input not found:', IN_CSV)
    print('Run train_autoencoder_selected.py first to produce reconstruction_selected.csv')
    sys.exit(1)

rows = []
with IN_CSV.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            r['reconstruction_mse'] = float(r.get('reconstruction_mse','nan'))
            r['is_anomaly'] = r.get('is_anomaly','').lower() in ('true','1','yes')
            rows.append(r)
        except Exception:
            continue

if not rows:
    print('No rows found in', IN_CSV)
    sys.exit(1)

# Overall stats
total = len(rows)
anom = sum(1 for r in rows if r['is_anomaly'])
percent = 100.0 * anom / total

# Per-dataset counts
counts = {}
for r in rows:
    d = r.get('dataset','')
    counts.setdefault(d, {'total':0, 'anom':0})
    counts[d]['total'] += 1
    if r['is_anomaly']:
        counts[d]['anom'] += 1

# Top anomalies by MSE
sorted_rows = sorted(rows, key=lambda r: r['reconstruction_mse'], reverse=True)
top_n = sorted_rows[:20]

# Write outputs
with OUT_SUM.open('w', encoding='utf-8') as f:
    f.write(f'Total samples: {total}\n')
    f.write(f'Anomalies: {anom} ({percent:.2f}% )\n\n')
    f.write('Per-dataset:\n')
    for d, v in sorted(counts.items()):
        pct = 100.0 * v['anom'] / v['total'] if v['total'] else 0.0
        f.write(f' - {d}: {v["anom"]}/{v["total"]} ({pct:.2f}%)\n')
    f.write('\nTop anomalies (by reconstruction_mse):\n')
    for r in top_n:
        f.write(f"{r.get('dataset')},{r.get('record_id')},{r.get('reconstruction_mse')},{r.get('is_anomaly')}\n")

with OUT_TOP.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['dataset','record_id','reconstruction_mse','is_anomaly'])
    writer.writeheader()
    for r in top_n:
        writer.writerow({'dataset':r.get('dataset'),'record_id':r.get('record_id'),'reconstruction_mse':r.get('reconstruction_mse'),'is_anomaly':r.get('is_anomaly')})

print('Wrote', OUT_SUM, 'and', OUT_TOP)

# Optional plot
try:
    import matplotlib.pyplot as plt
    labels = []
    anom_counts = []
    for d, v in sorted(counts.items()):
        labels.append(d)
        anom_counts.append(v['anom'])
    plt.figure(figsize=(6,4))
    plt.bar(labels, anom_counts, color='tab:blue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Anomaly count')
    plt.title('Anomalies by dataset')
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200)
    print('Saved plot:', OUT_PLOT)
except Exception:
    pass

print('Done.')
