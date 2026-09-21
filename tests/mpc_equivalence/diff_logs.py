import sys
import numpy as np
import pandas as pd

a = pd.read_csv(sys.argv[1]); b = pd.read_csv(sys.argv[2])
skip = {'t', 'dt', 'solve_ms', 'ts_ac', 'sim_dt', 'd_steps'}
print(f"rows: orig={len(a)} new={len(b)}")
cols_a, cols_b = set(a.columns), set(b.columns)
print("columns only in orig:", sorted(cols_a - cols_b)); print("columns only in new:", sorted(cols_b - cols_a))
n = min(len(a), len(b)); worst = []
for c in sorted(cols_a & cols_b):
    if c in skip: continue
    x, y = a[c].iloc[:n], b[c].iloc[:n]
    if x.dtype == object or y.dtype == object:
        neq = int((x.astype(str) != y.astype(str)).sum())
        if neq: worst.append((c, 'str-mismatch', neq))
        continue
    d = np.abs(x.values.astype(float) - y.values.astype(float))
    d = np.where(np.isnan(x.values.astype(float)) & np.isnan(y.values.astype(float)), 0.0, d)
    md = np.nanmax(d) if len(d) else 0.0
    if md > 0: worst.append((c, md, int(np.argmax(d))))
if not worst:
    print("IDENTICAL on all compared columns")
else:
    print("differences (col, max|diff|, first-argmax-row):")
    for w in sorted(worst, key=lambda t: -float(t[1]) if not isinstance(t[1], str) else 1e9): print("  ", w)
print("mode counts orig:", a['mode'].value_counts().to_dict(), " new:", b['mode'].value_counts().to_dict())
