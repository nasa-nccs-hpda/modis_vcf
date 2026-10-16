#!/usr/bin/env python3
"""Write a split whose predictor columns are recomputed, for the listed tiles, from a metrics root with the package's
Metrics.getMetricFromRf (for --year, predictor names keep their 2019 dates and are mapped to the same period of that year); all other
rows are copied from the base split. Train and Test are done in one pass per tile with one Metrics object. Also reports rows whose
predictors are all missing. Keys: x = raster row, y = column.
  --check : do not write a split; require the recomputed values to equal the base values exactly for every listed tile.
Usage: recompute_split_onepass.py --base <split dir> --predictors <file> --tiles <file> --root <metrics root> --year Y [-o <out dir> | --check]
Writes <out>/{xTrain,xTest}.parq (predictor columns only) and copies {y,keys}{Train,Test}.parq; <out>/recompute.json (changed cells per
tile and column). Refuses to overwrite."""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modis_vcf.model.Metrics import Metrics
from modis_vcf.model import LowSun

# --lowSun <le0|s70|off> / --lowSunCheck <on|off> (default le0; V4r inputs: off).
sys.argv = LowSun.configureFromArgv(sys.argv)
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
from modis_vcf.model.VcfPredict import MOD44_DIR

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--base', type=Path, required=True)
ap.add_argument('--predictors', type=Path, required=True)
ap.add_argument('--tiles', type=Path, required=True)
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('-o', type=Path)
ap.add_argument('--check', action='store_true')
ap.add_argument('--year', type=int, required=True)
a = ap.parse_args()
names = a.predictors.read_text().split()
tiles = a.tiles.read_text().split()
pt = ProductTypeMod44(MOD44_DIR)
log = logging.getLogger('recompute')
if not a.check:
    if a.o.exists():
        raise SystemExit('%s exists' % a.o)
    a.o.mkdir(parents=True)
report, ok = {}, True
X, K, TL = {}, {}, {}
for part in ('Train', 'Test'):
    X[part] = pd.read_parquet(a.base / ('x%s.parq' % part), columns=names)
    K[part] = pd.read_parquet(a.base / ('keys%s.parq' % part))
    TL[part] = K[part]['tid-year'].str[:6].to_numpy()
for t in tiles:
    m = None  # one Metrics per tile, shared by Train and Test, so its read cache serves both
    for part in ('Train', 'Test'):
        sel = np.flatnonzero(TL[part] == t)
        if len(sel) == 0:
            report.setdefault(t, {})[part] = 'no rows'
            continue
        if m is None:
            m = Metrics(t, a.year, pt, a.root, log)
        rr, cc = K[part].x.to_numpy()[sel], K[part].y.to_numpy()[sel]
        rep = {}
        for n in names:
            new = m.getMetricFromRf(n)[rr, cc].astype(X[part][n].dtype)
            old = X[part][n].to_numpy()[sel]
            same = (new == old) | (np.isnan(new) & np.isnan(old)) if np.issubdtype(new.dtype, np.floating) else (new == old)
            nd = int((~same).sum())
            if nd:
                rep[n] = nd
            if not a.check:
                X[part].iloc[sel, X[part].columns.get_loc(n)] = new
        report.setdefault(t, {})[part] = dict(rows=len(sel), changed_columns=len(rep), changed_cells=rep)
        if a.check and rep:
            ok = False
        print(t, part, len(sel), 'rows;', len(rep), 'columns changed', flush=True)
    del m
if a.check:
    print(json.dumps(dict(RESULT='PASS' if ok else 'FAIL', report=report)))
    sys.exit(0 if ok else 1)
for part in ('Train', 'Test'):
    allnan = int(X[part].isna().all(axis=1).sum())
    report.setdefault('_all_missing_rows', {})[part] = allnan
    print(part, 'rows with every predictor missing:', allnan, flush=True)
    X[part].to_parquet(a.o / ('x%s.parq' % part))
    for k in ('y', 'keys'):
        shutil.copyfile(a.base / ('%s%s.parq' % (k, part)), a.o / ('%s%s.parq' % (k, part)))
(a.o / 'recompute.json').write_text(json.dumps(report, indent=1))
