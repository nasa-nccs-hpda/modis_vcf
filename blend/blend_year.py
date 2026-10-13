#!/usr/bin/env python3
"""Blend A for one tile-year (tree layer, MODIS rows v06-v11 = 30N-30S):
 tree = floor((1 - w) * clear + w * partner + 0.5), w = cloud_w(CUR_ZS, heavy5) * cover_w(clear, ramp20-50) (gate_common);
        on pixels valid (0-100) in clear and partner; others keep clear. --mask <maps dir> also requires validity in that map's tree
        (the 2019 product used the V1-V4C map here).
 bare = min(stored bare, 100 - tree) (the product's sum cap; where tree went down and the old cap was binding the stored bare is a lower
        bound, counted as raw_needed); non-tree = 100 - tree - bare; SD layers and non-land codes copied.
--rawBare <dir>: <dir>/Percent_NonVegetated_Raw-<tile>-<Y>.tif (the bare model's uncapped value). Then bare = min(raw, 100 - tree)
        exactly (the VcfPredict sum cap), so raw_needed = 0. Self-test first: min(raw, 100 - clear tree) must equal the clear-side bare
        on every land pixel (exit 1 if not).
--check <tree tif>: the blended tree must equal it on pixels valid in both (exit 1 if not).
Writes <out>/<3 layers>-<tile>-<Y>.tif (+ SD layers copied when present; clear-side files as template) and <out>/<tile>.json.
Usage: blend_year.py --tile T --year Y --clear <maps dir> --partner <maps dir> --counts <dir with <tile>_counts_snow.npz> --out <dir>
       [--mask <maps dir>] [--check <tif>] [--rawBare <dir>]"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
sys.path.insert(0, str(Path(__file__).parent))
import gate_common as G  # noqa: E402
LAY = ['Percent_Tree_Cover', 'Percent_NonVegetated', 'Percent_NonTree_Vegetation', 'Percent_Tree_Cover_SD', 'Percent_NonVegetated_SD']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tile', required=True); ap.add_argument('--year', type=int, required=True)
    for k in ('clear', 'partner', 'counts', 'out'):
        ap.add_argument('--' + k, type=Path, required=True)
    ap.add_argument('--mask', type=Path); ap.add_argument('--check', type=Path); ap.add_argument('--rawBare', type=Path)
    a = ap.parse_args(); t, Y = a.tile, a.year
    nm = lambda lay: '%s-%s-%d.tif' % (lay, t, Y)
    rd = lambda d, lay: gdal.Open(str(d / nm(lay))).ReadAsArray()
    a.out.mkdir(parents=True, exist_ok=True)
    js = a.out / ('%s.json' % t)
    if js.exists():
        raise SystemExit('%s exists' % js)
    ok = lambda v: (v >= 0) & (v <= 100)
    tc0 = rd(a.clear, LAY[0]).astype(np.int16); bare0 = rd(a.clear, LAY[1]).astype(np.int16); nt0 = rd(a.clear, LAY[2])
    v0 = ok(tc0)
    part = rd(a.partner, LAY[0]).astype(np.float64); clear = tc0.astype(np.float64)
    valid = ok(clear) & ok(part)
    if a.mask:
        valid &= ok(rd(a.mask, LAY[0]).astype(np.float64))
    z = np.load(a.counts / ('%s_counts_snow.npz' % t))
    w = G.cloud_w(z['CUR_ZS'], 'heavy5') * G.cover_w(clear, None, 'ramp20-50')
    val = np.floor((1 - w) * clear + w * part + 0.5)
    tc1 = tc0.copy(); tc1[valid] = val[valid].astype(np.int16)
    vb = v0 & ok(bare0)
    down = vb & (tc1 < tc0); binding = vb & (bare0 == 100 - tc0)
    bare1 = bare0.copy(); bare1[vb] = np.minimum(bare0[vb], 100 - tc1[vb])
    raw_used = False
    if a.rawBare:
        raw = gdal.Open(str(a.rawBare / nm('Percent_NonVegetated_Raw'))).ReadAsArray().astype(np.int16)
        bad = int((vb & (np.minimum(raw, 100 - tc0) != bare0)).sum())
        if bad:
            raise SystemExit('%s: raw bare self-test failed on %d px (min(raw, 100 - clear tree) != clear bare)' % (t, bad))
        bare1[vb] = np.minimum(raw[vb], 100 - tc1[vb]); raw_used = True
    nt1 = nt0.astype(np.int16).copy(); nt1[vb] = 100 - tc1[vb] - bare1[vb]
    assert (nt1[vb] >= 0).all() and (tc1[vb] + bare1[vb] + nt1[vb] == 100).all()
    ch = v0 & (tc1 != tc0)
    res = dict(tile=t, year=Y, land=int(v0.sum()), w_pos=int((v0 & (w > 0)).sum()), tree_changed=int(ch.sum()),
               tree_up=int((v0 & (tc1 > tc0)).sum()), tree_down=int((v0 & (tc1 < tc0)).sum()),
               mean_change_where_changed=round(float((tc1[ch] - tc0[ch]).mean()), 3) if ch.any() else 0.0,
               mean_change_land=round(float((tc1[v0] - tc0[v0]).mean()), 4) if v0.any() else 0.0,
               bare_changed=int((vb & (bare1 != bare0)).sum()), raw_needed=0 if raw_used else int((down & binding).sum()), raw_bare=raw_used,
               unchanged_where_w0=bool((tc1[v0 & (w == 0)] == tc0[v0 & (w == 0)]).all()))
    if a.check:
        ref = gdal.Open(str(a.check)).ReadAsArray().astype(np.int16); both = v0 & ok(ref)
        res['check_differ'] = int((tc1[both] != ref[both]).sum()); res['check_compared'] = int(both.sum())
    for lay, arr in ((LAY[0], tc1), (LAY[1], bare1), (LAY[2], nt1)):
        ds = gdal.GetDriverByName('GTiff').CreateCopy(str(a.out / nm(lay)), gdal.Open(str(a.clear / nm(lay))), options=['COMPRESS=LZW'])
        ds.GetRasterBand(1).WriteArray(arr.astype(ds.GetRasterBand(1).ReadAsArray(0, 0, 1, 1).dtype)); ds.FlushCache(); ds = None
    for lay in LAY[3:]:
        if (a.clear / nm(lay)).exists():
            shutil.copyfile(a.clear / nm(lay), a.out / nm(lay))
    js.write_text(json.dumps(res)); print(json.dumps(res), flush=True)
    return 0 if res['unchanged_where_w0'] and res.get('check_differ', 0) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
