#!/usr/bin/env python3
"""DEPRECATED since v1.4.1: the uncapped non-vegetated value is used inside the map step. Kept for one version (to reproduce v1.4.0 runs); to be removed in v1.5.0.
The UNCAPPED non-vegetated layer for one tile-year, from the package's own prediction (VcfPredict._runPrediction, MEDIAN, water
masked), i.e. VcfPredict.runTileForYear without the sum cap. blend_year.py --rawBare and bp10_apply.py read it.
Built-in self-test (every tile): min(raw, 100 - stored tree) on land must equal the stored capped bare map on every pixel; otherwise
exit 1. Run in low-sun mode le0 (the default) with the v1.4.0 models. Not for v00 tiles.
Writes <out>/Percent_NonVegetated_Raw-<tile>-<year>.tif and <out>/<tile>.json.
Usage: rawbare.py <tile> <tree model dir> <bare model dir> <metrics root> <stored map dir> <out dir> --year Y"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal

from modis_vcf.model.VcfPredict import VcfPredict

ap = argparse.ArgumentParser(description=__doc__)
for k in ('tile', 'tdir', 'bdir', 'root', 'maps', 'out'):
    ap.add_argument(k)
ap.add_argument('--year', type=int, required=True)
a = ap.parse_args()
t, y = a.tile, a.year
tdir, bdir, root, maps, out = Path(a.tdir), Path(a.bdir), Path(a.root), Path(a.maps), Path(a.out)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
assert VcfPredict._vTile(t) not in VcfPredict.HIGH_LATITUDE_BARE_VTILES
# The bare model stands in for the tree model (never used here), halving memory; tdir is ignored.
vp = VcfPredict(bdir / 'pctbare.bin', bdir / 'pctbareTopN.bin', bdir / 'pctbare.bin', bdir / 'pctbareTopN.bin', out, root,
                nJobs=8, treeCoverAgg='median', nonvegAgg='median', writeSd=False)
raw, _ = vp._runPrediction(t, y, vp._nonvegRf, vp._nonvegTopN, vp._nonvegAgg)
mask = vp._getMask(t, y)
ok = (mask != VcfPredict.MASK_WATER) & (mask != VcfPredict.MASK_OUT_OF_PROJ)
tree = gdal.Open(str(maps / ('Percent_Tree_Cover-%s-%d.tif' % (t, y)))).ReadAsArray().astype(np.int32)
bare = gdal.Open(str(maps / ('Percent_NonVegetated-%s-%d.tif' % (t, y)))).ReadAsArray().astype(np.int32)
r = raw.astype(np.int32)
capped = np.where(ok, np.minimum(r, 100 - tree), r)
ndiff = int((capped != bare).sum())
land = ok & (r >= 0) & (r <= 100)
res = dict(tile=t, year=y, bare_model=str(bdir), metrics=str(root), selftest_pixels_differ=ndiff, land=int(land.sum()),
           cap_binding=int((land & (r > 100 - tree)).sum()), raw_mean=float(r[land].mean()), stored_mean=float(bare[land].mean()))
vp._write(t, y, 'Percent_NonVegetated_Raw', raw)
(out / ('%s.json' % t)).write_text(json.dumps(res))
print(json.dumps(res), flush=True)
sys.exit(0 if ndiff == 0 else 1)
