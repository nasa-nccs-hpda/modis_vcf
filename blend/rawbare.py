#!/usr/bin/env python3
"""The UNCAPPED non-vegetated layer for one tile (2019), from the package's own prediction (VcfPredict._runPrediction, MEDIAN, water
masked), i.e. VcfPredict.runTileForYear without the sum cap. blend_year.py --rawBare and bp10_apply.py read it.
Built-in self-test (every tile): min(raw, 100 - stored tree) on land must equal the stored capped bare map on every pixel; otherwise
exit 1. Run with the v1.4.0 (LE0) environment and models. Not for v00 tiles. 2019 only (the year is fixed in _runPrediction).
Writes <out>/Percent_NonVegetated_Raw-<tile>-2019.tif and <out>/<tile>.json.
Usage: rawbare.py <tile> <tree model dir> <bare model dir> <metrics root> <stored map dir> <out dir>"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal

from modis_vcf.model.VcfPredict import VcfPredict

t, tdir, bdir, root, maps, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6])
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
assert VcfPredict._vTile(t) not in VcfPredict.HIGH_LATITUDE_BARE_VTILES
# The bare model stands in for the tree model (never used here), halving memory; tdir is ignored.
vp = VcfPredict(bdir / 'pctbare.bin', bdir / 'pctbareTopN.bin', bdir / 'pctbare.bin', bdir / 'pctbareTopN.bin', out, root,
                nJobs=8, treeCoverAgg='median', nonvegAgg='median', writeSd=False)
raw, _ = vp._runPrediction(t, 2019, vp._nonvegRf, vp._nonvegTopN, vp._nonvegAgg)
mask = vp._getMask(t, 2019)
ok = (mask != VcfPredict.MASK_WATER) & (mask != VcfPredict.MASK_OUT_OF_PROJ)
tree = gdal.Open(str(maps / ('Percent_Tree_Cover-%s-2019.tif' % t))).ReadAsArray().astype(np.int32)
bare = gdal.Open(str(maps / ('Percent_NonVegetated-%s-2019.tif' % t))).ReadAsArray().astype(np.int32)
r = raw.astype(np.int32)
capped = np.where(ok, np.minimum(r, 100 - tree), r)
ndiff = int((capped != bare).sum())
land = ok & (r >= 0) & (r <= 100)
res = dict(tile=t, bare_model=str(bdir), metrics=str(root), selftest_pixels_differ=ndiff, land=int(land.sum()),
           cap_binding=int((land & (r > 100 - tree)).sum()), raw_mean=float(r[land].mean()), stored_mean=float(bare[land].mean()))
vp._write(t, 2019, 'Percent_NonVegetated_Raw', raw)
(out / ('%s.json' % t)).write_text(json.dumps(res))
print(json.dumps(res), flush=True)
sys.exit(0 if ndiff == 0 else 1)
