#!/usr/bin/env python3
"""Rebuild a tile-year's 1-Days and standard composites with the package pipeline (Metrics.getBandCube for the 8 bands), into a
fresh root. Check: every composite .bin and its -qa.npy / -cloudfb.npy must equal the stored ones (<ref root>, same layout) byte for byte;
the 1-Days are then the inputs those composites were made from. Exit 1 on any difference.
Usage: days_year.py <tile> <year> <out root> <ref root>"""
import filecmp
import json
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

t, y, root, ref = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
BANDS = ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7', 'Band_31']
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
root.mkdir(parents=True, exist_ok=True)
if (root / t / str(y)).exists():
    raise SystemExit('exists: %s' % (root / t / str(y)))
t0 = time.time()
m = Metrics(t, y, ProductTypeMod44(MOD44_DIR), root, logging.getLogger('days'))
for b in BANDS:
    m.getBandCube(b)
new = root / t / str(y) / '2-Composites'; old = ref / t / str(y) / '2-Composites'
files = sorted(p.name for p in old.iterdir() if p.name.endswith(('.bin', '-qa.npy', '-cloudfb.npy')) and '-Band_' in p.name)
bad = [f for f in files if not (new / f).exists() or not filecmp.cmp(new / f, old / f, shallow=False)]
ndays = len(list((root / t / str(y) / '1-Days').iterdir()))
rep = dict(tile=t, year=y, compared=len(files), differ=len(bad), first_differ=bad[:5], day_files=ndays, seconds=round(time.time() - t0))
(root / t / str(y) / 'days_check.json').write_text(json.dumps(rep))
print(json.dumps(rep), flush=True)
sys.exit(0 if files and not bad else 1)
