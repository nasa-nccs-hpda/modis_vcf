#!/usr/bin/env python3
"""V4r composites and snow-aware cloud counts for one tile-year (blend A inputs).
Slots = DOY 065-353 of Y and 001-049 of Y+1 (23 inputs) + the never-read 24th slot 065 of Y+1, as the pipeline's days sought.
MOD44CQ is read from /css/modis/Collection6.1/L3/MOD44B-VCF/dev/<slot year>/ (the folder the pipeline reads). The shadow/water gate
ratio (< 0.6) is reported, not fatal. Land / dense statistics use the year's own tree map (--land) and C6 of the year when present.
Composites: where all present MOD44CQ inputs of a 48-day period are cloud or cloud-adjacent (state bits 0-1 != 0 or bit 6) and the period
fell back (qa.npy = 1), all 8 bands are replaced by the input with the lowest red among inputs present, not shadow (bit 2), valid,
NDVI >= 0 and NIR >= 1500. Counts: the same trigger plus some input with 0 < zenith < 65; CUR_Z, CUR_ZS (NDSI snow pick not counted:
(B4-B6)/(B4+B6) >= 0.4 and B2 >= 1100 on the V4r composite), CUR_ZB (state bit 5 of the pick not counted).
Checks (exit 3): alignment (the stored all-input mean equals the composite at fallback land pixels, >= 0.999); the pick's Band_1 equals
the V4r Band_1 at every counted pixel-period.
Writes <out>/<tile>/<Y>/2-Composites (V4r, with qa/cloud .npy copied), <out>/<tile>/<Y>/1-Days -> <days>/1-Days, and in --statsDir:
<tile>_v4r_used_count.npy, <tile>_counts_snow.npz, <tile>_compose.json.
Usage: v4r_year.py --tile hXXvYY --year Y --days <days root>/<tile>/<Y> --out <V4r root> --statsDir <dir> --land <tree map tif>"""
import argparse, glob, json, os, shutil, sys
from pathlib import Path
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
DEV = '/css/modis/Collection6.1/L3/MOD44B-VCF/dev/'
BANDS = ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7', 'Band_31']
NIR_FLOOR = 1500
key = lambda yd: '%d%03d' % yd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tile', required=True)
    ap.add_argument('--year', type=int, required=True)
    for k in ('days', 'out', 'statsDir', 'land'):
        ap.add_argument('--' + k, type=Path, required=True)
    a = ap.parse_args()
    T, Y, SRC = a.tile, a.year, a.days
    SLOTS = [(Y, d) for d in range(65, 366, 16)] + [(Y + 1, d) for d in (1, 17, 33, 49)] + [(Y + 1, 65)]
    assert len(SLOTS) == 24
    OUT = a.out / T / str(Y)
    if OUT.exists():
        raise SystemExit('output exists, refusing: %s' % OUT)
    (OUT / '2-Composites').mkdir(parents=True)
    os.symlink(SRC / '1-Days', OUT / '1-Days')
    a.statsDir.mkdir(parents=True, exist_ok=True)

    def sub(yd, i):
        if yd == SLOTS[-1]:
            return None
        fs = glob.glob(DEV + '%d/MOD44CQ.A%s.%s.061.*.hdf' % (yd[0], key(yd), T))
        return gdal.Open(gdal.Open(sorted(fs)[-1]).GetSubDatasets()[i][0]).ReadAsArray() if fs else None

    def day(yd, b):
        f = SRC / '1-Days' / ('MOD44-%s-%s-%s.bin' % (T, key(yd), b))
        return np.fromfile(f, np.int16).reshape(4800, 4800) if f.exists() else None

    H = None
    hf = sorted(glob.glob('/css/modis/Collection6/L3/MOD44B-VCF/%d/065/MOD44B.A%d065.%s.006.*.hdf' % (Y, Y, T)))
    if hf:
        for n, _ in gdal.Open(hf[0]).GetSubDatasets():
            if n.replace('"', '').endswith(':Percent_Tree_Cover'):
                H = gdal.Open(n).ReadAsArray().astype(int)
    G = gdal.Open(str(a.land)).ReadAsArray()
    land = (G <= 100) & ((H <= 100) if H is not None else True)
    dense = land & (H >= 60) if H is not None else np.zeros_like(land)
    sets = dict(land=land, dense=dense)
    stats, align = [], []
    nused = np.zeros((4800, 4800), np.uint8)
    C = {k: np.zeros((4800, 4800), np.uint8) for k in ('CUR_Z', 'CUR_ZS', 'CUR_ZB')}
    per, pick_mism = [], 0
    for p in range(8):
        slots = SLOTS[3 * p: 3 * p + 3]; pk = key(slots[0])
        st = [sub(s, 0) for s in slots]
        avail = [s is not None for s in st]
        cloudy = np.ones((4800, 4800), bool); zen_ok = np.zeros((4800, 4800), bool)
        for i, s in enumerate(st):
            if s is not None:
                cloudy &= ((s & 3) != 0) | ((s & 64) != 0)
                z = sub(slots[i], 2).astype(np.int32); zen_ok |= (z > 0) & (z < 65)
        qa = np.load(SRC / '2-Composites' / ('MOD44-%s-%s-Band_1-qa.npy' % (T, pk))).astype(bool)
        trig = cloudy & qa & any(avail)
        b1 = [day(s, 'Band_1') for s in slots]; b2 = [day(s, 'Band_2') for s in slots]
        red = np.full((len(slots), 4800, 4800), np.iinfo(np.int32).max, np.int64)
        for i in range(len(slots)):
            if st[i] is None or b1[i] is None:
                continue
            r, n = b1[i].astype(np.int64), b2[i].astype(np.int64)
            sh = (st[i] & 4) != 0
            inv = (r <= -100) | (n <= -100) | (r >= 10000) | (n >= 10000)
            nd = np.where(r + n != 0, (n - r) / np.where(r + n != 0, r + n, 1), -1.0)
            ok = ~sh & ~inv & ~(nd < 0) & ~(n < NIR_FLOOR)
            red[i] = np.where(ok, r, red[i])
        anyok = red.min(0) < np.iinfo(np.int32).max
        pick = red.argmin(0)
        use = trig & anyok
        nused += use
        rec = dict(period=pk, inputs_available=int(sum(avail)))
        comps = {}
        for b in BANDS:
            old = np.fromfile(SRC / '2-Composites' / ('MOD44-%s-%s-%s.bin' % (T, pk, b)), np.int16).reshape(4800, 4800)
            ins = [day(s, b) for s in slots]
            present = [x if x is not None else np.full((4800, 4800), 10000, np.int16) for x, sl in zip(ins, slots) if x is not None or sl != SLOTS[-1]]
            fbm = qa & land
            if present:
                allmean = np.mean(np.stack(present).astype(np.float64), 0).astype(np.int16)
                align.append(dict(period=pk, band=b, fallback_px=int(fbm.sum()),
                                  identical=float((allmean[fbm] == old[fbm]).mean()) if fbm.any() else 1.0))
            stack = np.stack([x if x is not None else np.zeros((4800, 4800), np.int16) for x in ins])
            picked = np.take_along_axis(stack, pick[None], 0)[0]
            np.where(use, picked, old).astype(np.int16).tofile(OUT / '2-Composites' / ('MOD44-%s-%s-%s.bin' % (T, pk, b)))
            for suf in ('-qa.npy', '-cloudfb.npy'):
                s = SRC / '2-Composites' / ('MOD44-%s-%s-%s%s' % (T, pk, b, suf))
                if s.exists():
                    shutil.copy(s, OUT / '2-Composites')
            comps[b] = (old, np.where(use, picked, old))
        o1, n1 = comps['Band_1']; o2, n2 = comps['Band_2']
        ndv = lambda r, n: (n.astype(float) - r) / np.maximum(n.astype(float) + r, 1)
        for sname, m in sets.items():
            u = use & m
            rec[sname] = dict(px=int(m.sum()), trig_share=float((trig & m).sum() / max(1, m.sum())), used_share=float(u.sum() / max(1, m.sum())),
                              used_px=int(u.sum()),
                              new_med=dict(red=float(np.median(n1[u])), nir=float(np.median(n2[u])), ndvi=float(np.median(ndv(n1[u], n2[u])))) if u.any() else None,
                              clear_ref=dict(nir=float(np.median(o2[m & ~qa]))) if (m & ~qa).any() else None)
        stats.append(rec)
        # counts (counts_snow.py): zenith-gated trigger; snow tests on the pick
        cu = use & zen_ok
        stackb1 = np.stack([x if x is not None else np.zeros((4800, 4800), np.int16) for x in b1])
        pick_mism += int((np.take_along_axis(stackb1, pick[None], 0)[0][cu] != comps['Band_1'][1][cu]).sum())
        c2, c4, c6 = (comps[b][1].astype(np.float64) for b in ('Band_2', 'Band_4', 'Band_6'))
        ndsi = (c4 - c6) / np.where(c4 + c6 != 0, c4 + c6, 1)
        snow_ndsi = (ndsi >= 0.4) & (c2 >= 1100) & (c4 + c6 != 0)
        sstack = np.stack([s if s is not None else np.zeros((4800, 4800), np.uint16) for s in st]).astype(np.int64)
        snow_bit = (np.take_along_axis(sstack, pick[None], 0)[0] & 32) != 0
        C['CUR_Z'] += cu; C['CUR_ZS'] += cu & ~snow_ndsi; C['CUR_ZB'] += cu & ~snow_bit
        u = cu & land; nn = max(1, int(u.sum()))
        per.append(dict(period=pk, curz_share=float(u.sum() / max(1, land.sum())), ndsi_snow_share_of_curz=float((u & snow_ndsi).sum() / nn),
                        bit_snow_share_of_curz=float((u & snow_bit).sum() / nn)))
        print(json.dumps(rec), flush=True)
    np.save(a.statsDir / ('%s_v4r_used_count.npy' % T), nused)
    np.savez_compressed(a.statsDir / ('%s_counts_snow.npz' % T), **C)
    worst = min([x['identical'] for x in align] or [1.0])
    ratios = [s['dense']['new_med']['nir'] / s['dense']['clear_ref']['nir'] for s in stats
              if s['dense']['new_med'] and s['dense']['clear_ref'] and s['dense']['used_px'] >= 10000]
    gate = float(np.median(ratios)) if ratios else None
    summary = dict(tile=T, year=Y, alignment_min_identical=worst, gate_nir_ratio=gate, gate_flag=bool(gate is not None and gate < 0.6),
                   pick_band1_mismatch=pick_mism,
                   land_pixel_periods_replaced_share=float(nused[land].sum() / max(1, 8 * land.sum())),
                   land_pixels_any_replaced_share=float((nused[land] > 0).mean()) if land.any() else 0.0,
                   land_dist={k: np.bincount(v[land], minlength=9).tolist() for k, v in C.items()})
    (a.statsDir / ('%s_compose.json' % T)).write_text(json.dumps(dict(summary=summary, stats=stats, align=align, periods=per), indent=1))
    print(json.dumps(summary), flush=True)
    if worst < 0.999:
        print('ALIGNMENT FAILED'); return 3
    if pick_mism:
        print('PICK MISMATCH'); return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
