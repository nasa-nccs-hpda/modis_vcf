"""Terra composites with Aqua good days where Terra has none (the MCD44B composites), one tile and period.

Ported unchanged from the Aqua-Terra Experiment's ta_merge.py (Aqua_Fill_Fallback_Plan.md; lead 2026-11-18: D1 (a) no
offset adjustment, D2 (a) fill only where Aqua Band 6 is valid; adopted for v2.0.0 on 2026-11-19).  Per pixel, from the
finished Terra and Aqua DailiesComposite outputs of the period (no dailies are read):

  Terra fbclass 0 (good)                                     -> Terra, unchanged
  Terra fbclass 1-5 and Aqua fbclass 0 and Aqua b6_n > 0     -> Aqua, all 8 bands (one sensor per pixel-period); class 7
  otherwise                                                  -> Terra, unchanged (Aqua's own fallback is never used)

qa / cloudfb of class 7 = 0 (Aqua had a good day).  Diag: fbclass (7 added), fill_src (uint8, 1 = Aqua; the source of the
product's Aqua_Fill layer), solz_used / vz_used / gooddays / group (Aqua's at class 7), summary.json.  The other Terra diag
files are not carried.  With aquaDir None (no Aqua dailies in the period) the output is the Terra composite, fill_src all 0.

b6rule 'b' (off by default): no Band 6 test; Band 6 stays Terra's where Aqua b6_n == 0.
"""

import os
import time

import numpy as np

from modis_vcf.model.DailiesComposite import BANDS, N, compositeName, diagName, _writeJson

FILL_CLASS = 7


def merge(tile: str, period: str, terraDir: str, aquaDir: str, out: str, b6rule: str = 'a', diag: bool = False) -> dict:
    """Write the merged composites of one period into out; returns the summary (also in diag).  fbclass and fill_src are
    always written; diag adds solz_used / vz_used / gooddays / group (the inputs must then have been built with diag)."""
    if b6rule not in ('a', 'b'):
        raise ValueError('b6rule must be a or b')

    t0 = time.time()
    dg = lambda d, k: np.load(diagName(d, tile, period, k))
    ct = dg(terraDir, 'fbclass')

    if aquaDir is None:
        fill = np.zeros(ct.shape, bool)
    else:
        ca = dg(aquaDir, 'fbclass')
        assert ct.shape == ca.shape == (N, N) and ct.dtype == np.int8, (ct.shape, ca.shape, ct.dtype)
        fill = (ct > 0) & (ca == 0)

        if b6rule == 'a':
            fill &= dg(aquaDir, 'b6_n') > 0

    os.makedirs(out + '/2-Composites', exist_ok=True)
    os.makedirs(out + '/diag', exist_ok=True)
    rd = lambda d, b: np.fromfile(compositeName(d, tile, period, b), np.int16).reshape(N, N)

    for b in BANDS:
        tv = rd(terraDir, b)

        if b == 'Band_6' and b6rule == 'b' and aquaDir is not None:
            v = np.where(fill & (dg(aquaDir, 'b6_n') > 0), rd(aquaDir, b), tv)
        else:
            v = np.where(fill, rd(aquaDir, b), tv) if fill.any() else tv

        v.astype(np.int16).tofile(compositeName(out, tile, period, b))

        for k in ('qa', 'cloudfb'):
            q = np.load(compositeName(terraDir, tile, period, b, '-%s.npy' % k))
            np.save(compositeName(out, tile, period, b, '-%s.npy' % k), np.where(fill, 0, q).astype(q.dtype))

    dn = lambda k: diagName(out, tile, period, k)
    np.save(dn('fbclass'), np.where(fill, FILL_CLASS, ct).astype(ct.dtype))
    np.save(dn('fill_src'), fill.astype(np.uint8))

    for k in ('solz_used', 'vz_used', 'gooddays', 'group') if diag else ():
        t = dg(terraDir, k)
        np.save(dn(k), np.where(fill, dg(aquaDir, k), t).astype(t.dtype) if aquaDir is not None else t)

    cls = np.where(fill, FILL_CLASS, ct)
    s = dict(tile=tile, period=period, rules='Terra B2 + Aqua fill', b6rule=b6rule, terra=terraDir, aqua=aquaDir,
             diag=diag, fill_share=float(fill.mean()), terra_fallback_share=float((ct > 0).mean()),
             fbclass_share={int(k): float((cls == k).mean()) for k in range(8)}, seconds=round(time.time() - t0, 1))
    _writeJson(dn('summary.json'), s)
    return s
