#!/usr/bin/env python3
"""Acceptance test of a year's V4r split against the year's standard split (same keys and labels by construction).
At rows whose pixel V4r never replaced (used count 0; keys x = raster row, y = column) every predictor must equal the standard value
(NaN = NaN). Reports the share of rows replaced and of their cells changed.
Usage: accept_split.py <v4r split> <standard split> <usedDir> <out json>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sp, ref, ud, out = map(Path, sys.argv[1:5])
rep, ok = {}, True
for part in ('Train', 'Test'):
    K = pd.read_parquet(ref / ('keys%s.parq' % part)); Kn = pd.read_parquet(sp / ('keys%s.parq' % part))
    same_keys = K.equals(Kn); ok &= same_keys
    Xn = pd.read_parquet(sp / ('x%s.parq' % part)); Xo = pd.read_parquet(ref / ('x%s.parq' % part), columns=list(Xn.columns))
    tl = K['tid-year'].str[:6].to_numpy(); used = np.zeros(len(K), np.uint8)
    for t in np.unique(tl):
        m = tl == t; u = np.load(ud / ('%s_v4r_used_count.npy' % t)); used[m] = u[K.x.to_numpy()[m], K.y.to_numpy()[m]]
    a, b = Xn.to_numpy(np.float64), Xo.to_numpy(np.float64)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    bad0 = int((~same[used == 0]).sum()); ok &= bad0 == 0
    leq = bool(pd.read_parquet(ref / ('y%s.parq' % part)).equals(pd.read_parquet(sp / ('y%s.parq' % part)))); ok &= leq
    rep[part] = dict(rows=len(K), keys_equal=bool(same_keys), labels_equal=leq, rows_replaced=int((used > 0).sum()),
                     share_rows_replaced=float((used > 0).mean()), cells_differ_where_unreplaced=bad0,
                     share_cells_changed_where_replaced=float((~same[used > 0]).mean()) if (used > 0).any() else 0.0)
rep['RESULT'] = 'PASS' if ok else 'FAIL'
out.write_text(json.dumps(rep, indent=1)); print(json.dumps(rep)); sys.exit(0 if ok else 1)
