#!/usr/bin/env python3
"""Compute (and cache in the root's 3-Metrics) every metric the tree and bare predictor lists need, for one tile-year, so
later readers never race on a cache file.
Usage: warm_metrics.py <tile> <year> <root> <predictor file> [<predictor file> ...]"""
import logging
import sys
import time
from pathlib import Path

from modis_vcf.model.Metrics import Metrics
from modis_vcf.model import LowSun

# --lowSun <le0|s70|off> / --lowSunCheck <on|off> (default le0; V4r inputs: off).
sys.argv = LowSun.configureFromArgv(sys.argv)
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
from modis_vcf.model.VcfPredict import MOD44_DIR

t, year, root = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
names = []
for f in sys.argv[4:]:
    names += [n for n in open(f).read().split() if n not in names]
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
m = Metrics(t, year, ProductTypeMod44(MOD44_DIR), root, logging.getLogger('warm'))
print('days sought:', m._daysSought, flush=True)
t0 = time.time()
for n in names:
    m.getMetricFromRf(n)
print('%s: %d predictors cached in %d s' % (t, len(names), time.time() - t0), flush=True)
