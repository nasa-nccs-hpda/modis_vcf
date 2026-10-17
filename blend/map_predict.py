#!/usr/bin/env python3
"""DEPRECATED since v1.4.1: VcfPredict predicts the partner (--partner, --partnerTopN, --partnerMetricsDir). Kept for one version (to reproduce v1.4.0 runs); to be removed in v1.5.0.
Tree-cover map for one tile, one year and one model (the V4R60 partner of blend A).
Features are read from <metricsDir>/<tile>/<year>/3-Metrics/<family>.tif by band name (band metadata 'Description' with spaces -> '-',
e.g. 'CGreenMean13 NDVI' -> 'CGreenMean13-NDVI'). The model's predictor names carry 2019 dates; year_name() maps them to --year with the
rule of Metrics.getMetricFromRf / getYearForDay (DOY 65..353 -> year, DOY 33 -> year + 1); identity for 2019. --year is required.
fallback_count / cloudy_count come from --counts (<tile>_counts.npz) when the model uses them.
Value = MEDIAN of the trees (each clipped to 0-100), rounded (floor(v + 0.5)), as VcfPredict.
Non-land codes (water 200, other no-data) are copied from --ref, so only land pixels (0-100 in --ref) are predicted.
--check: the result must equal --ref on every pixel (with the product's own model this proves the feature assembly; exit 1 if not).
Writes <out>/Percent_Tree_Cover-<tile>-<year>.tif (Int16, no georeferencing, like the product GeoTIFFs).
Usage: map_predict.py --tile hXXvYY --year YYYY --model <dir> --metricsDir <root> --ref <maps dir> --out <dir> [--counts <dir>] [--check]"""
import argparse, json, pickle, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import joblib
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

# Metrics._daysSought (three sixteen-day composites): DOY -> year offset; mirrors Metrics.getYearForDay.
DAYS_SOUGHT = {65: 0, 113: 0, 161: 0, 209: 0, 257: 0, 305: 0, 353: 0, 33: 1}


def year_name(c, year):
    """Predictor name with its Day_yyyyddd mapped to the metrics of <year> (as Metrics.getMetricFromRf)."""
    parts = c.split('-')
    if len(parts) == 3 and parts[2].startswith('Day_') and len(parts[2]) == 11:
        doy = int(parts[2][8:])
        if doy not in DAYS_SOUGHT:
            raise SystemExit('predictor %s: day %d not in the composite schedule' % (c, doy))
        parts[2] = 'Day_%d%03d' % (year + DAYS_SOUGHT[doy], doy)
    return '-'.join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tile', required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--metricsDir', type=Path, required=True)
    ap.add_argument('--ref', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--counts', type=Path)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--threads', type=int, default=10)
    ap.add_argument('--year', type=int, required=True)
    a = ap.parse_args()
    T = a.tile
    t0 = time.time()
    names = list(pickle.load(open(a.model / 'pcttreeTopN.bin', 'rb')))
    ref = gdal.Open(str(a.ref / ('Percent_Tree_Cover-%s-%d.tif' % (T, a.year)))).ReadAsArray()
    land = (ref >= 0) & (ref <= 100)
    idx = np.flatnonzero(land)
    mdir = a.metricsDir / T / str(a.year) / '3-Metrics'
    X = np.empty((len(idx), len(names)), np.float32)
    bandmap, src = {}, {}
    cnt = np.load(a.counts / ('%s_counts.npz' % T)) if a.counts else None
    for j, c in enumerate(names):
        if cnt is not None and c in cnt.files:
            X[:, j] = cnt[c].ravel()[idx]; src[c] = 'counts'; continue
        fam = c.split('-')[0]
        if fam not in bandmap:
            ds = gdal.Open(str(mdir / ('%s.tif' % fam)))
            bandmap[fam] = (ds, {ds.GetRasterBand(i).GetMetadata().get('Description', '').replace(' ', '-'): i for i in range(1, ds.RasterCount + 1)})
        ds, bm = bandmap[fam]
        cy = year_name(c, a.year)
        if cy not in bm:
            raise SystemExit('%s: no band %s (model name %s) in %s.tif' % (T, cy, c, fam))
        X[:, j] = ds.GetRasterBand(bm[cy]).ReadAsArray().ravel()[idx]; src[c] = '%s.tif band %d (%s)' % (fam, bm[cy], cy)
    del bandmap
    # A compact copy (<model>/pcttree.cf, model/CompactForest.py) is used when present (identical predictions).
    if (a.model / 'pcttree.cf').is_dir():
        from modis_vcf.model.CompactForest import CompactForest
        model = CompactForest.load(a.model / 'pcttree.cf')
    else:
        model = joblib.load(a.model / 'pcttree.bin')
    v = np.empty(len(idx), np.float64)
    step = 2_000_000
    with ThreadPoolExecutor(a.threads) as ex:
        for s in range(0, len(idx), step):
            xs = X[s:s + step]
            P = np.vstack(list(ex.map(lambda e: np.clip(e.predict(xs), 0, 100), model.estimators_)))
            v[s:s + step] = np.median(P, axis=0)
    out = ref.astype(np.int16).copy()
    out.ravel()[idx] = np.clip(np.floor(v + 0.5), 0, 100).astype(np.int16)
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / ('Percent_Tree_Cover-%s-%d.tif' % (T, a.year))
    d = gdal.GetDriverByName('GTiff').Create(str(f), 4800, 4800, 1, gdal.GDT_Int16, options=['COMPRESS=LZW'])
    d.GetRasterBand(1).WriteArray(out); d = None
    rep = dict(tile=T, model=str(a.model), land_px=int(len(idx)), seconds=round(time.time() - t0))
    if a.check:
        nd = int((out != ref).sum()); rep['differs_from_ref_px'] = nd; rep['RESULT'] = 'PASS' if nd == 0 else 'FAIL'
    (a.out / ('%s_predict.json' % T)).write_text(json.dumps(dict(rep, features=src), indent=1))
    print(json.dumps(rep), flush=True)
    return 1 if a.check and rep['RESULT'] != 'PASS' else 0


if __name__ == '__main__':
    sys.exit(main())
