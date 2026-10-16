#!/usr/bin/env python3
"""BP10, the last step of v1.4.0 (all tiles): where the bare model's raw value is 100 and the final tree is 1-9: tree = 0, bare = 100,
non-tree = 0. Elsewhere the input is copied.
Input: <src>/<layer>-<tile>-<year>.tif (blend_year.py output for the blend tiles, the LE0 maps for the others), raw bare
<raw>/Percent_NonVegetated_Raw-<tile>-<year>.tif (rawbare.py; first --raw dir that has the tile). Land = all three layers 0-100; other
codes are copied. Self-test before any change: on land, min(raw, 100 - tree) must equal the input bare (the sum cap), else exit 1.
No raw file (v00 tiles): layers copied unchanged; the json reports where BP10 could have acted (tree 1-9 and bare = 100 - tree).
Writes <out>/<3 layers>-<tile>-<year>.tif and <json>/<tile>.json.
Usage: bp10_apply.py --tile T --year Y --src <dir> --out <dir> --json <dir> [--raw <dir> ...]"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
LAY = ['Percent_Tree_Cover', 'Percent_NonVegetated', 'Percent_NonTree_Vegetation']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tile', required=True); ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--json', type=Path, required=True)
    ap.add_argument('--raw', type=Path, nargs='*', default=[])
    a = ap.parse_args(); t = a.tile
    nm = lambda lay: '%s-%s-%d.tif' % (lay, t, a.year)
    js = a.json / ('%s.json' % t)
    if js.exists() or any((a.out / nm(l)).exists() for l in LAY):
        raise SystemExit('%s: output exists' % t)
    tc, ba, nt = (gdal.Open(str(a.src / nm(l))).ReadAsArray().astype(np.int16) for l in LAY)
    land = (tc >= 0) & (tc <= 100) & (ba >= 0) & (ba <= 100) & (nt >= 0) & (nt <= 100)
    assert (tc[land] + ba[land] + nt[land] == 100).all(), t
    rawf = [d / nm('Percent_NonVegetated_Raw') for d in a.raw if (d / nm('Percent_NonVegetated_Raw')).exists()]
    res = dict(tile=t, src=str(a.src), land=int(land.sum()), raw=str(rawf[0]) if rawf else None)
    if not rawf:
        res.update(bp10=0, bp10_upper_bound_no_raw=int((land & (tc >= 1) & (tc <= 9) & (ba == 100 - tc)).sum()))
        for l in LAY:
            shutil.copyfile(a.src / nm(l), a.out / nm(l))
    else:
        raw = gdal.Open(str(rawf[0])).ReadAsArray().astype(np.int16)
        bad = int((land & (np.minimum(raw, 100 - tc) != ba)).sum())
        if bad:
            raise SystemExit('%s: sum-cap self-test failed on %d px' % (t, bad))
        bp = land & (raw == 100) & (tc >= 1) & (tc <= 9)
        res.update(bp10=int(bp.sum()), tree_removed_sum=int(tc[bp].sum()),
                   bp_ge10_left=int((land & (raw == 100) & (tc >= 10)).sum()),
                   by_tree={str(v): int((bp & (tc == v)).sum()) for v in range(1, 10)})
        tc2, ba2, nt2 = tc.copy(), ba.copy(), nt.copy(); tc2[bp] = 0; ba2[bp] = 100; nt2[bp] = 0
        assert (tc2[land] + ba2[land] + nt2[land] == 100).all()
        for l, arr in zip(LAY, (tc2, ba2, nt2)):
            src = gdal.Open(str(a.src / nm(l)))
            ds = gdal.GetDriverByName('GTiff').CreateCopy(str(a.out / nm(l)), src, options=['COMPRESS=LZW'])
            dt = src.GetRasterBand(1).ReadAsArray(0, 0, 1, 1).dtype
            ds.GetRasterBand(1).WriteArray(arr.astype(dt)); ds.FlushCache(); ds = None
            chk = gdal.Open(str(a.out / nm(l))).ReadAsArray().astype(np.int16)
            assert (chk == arr).all(), (t, l, 'written != intended')
            assert ((chk != gdal.Open(str(a.src / nm(l))).ReadAsArray().astype(np.int16)) <= bp).all(), (t, l, 'change outside BP10')
    js.write_text(json.dumps(res)); print(json.dumps(res), flush=True)


if __name__ == '__main__':
    sys.exit(main())
