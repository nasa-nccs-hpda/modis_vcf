#!/usr/bin/env python3
"""Pre-flight for a new year: every predictor of the partner model resolves (map_predict.year_name) to a band in the year's
V4r metrics, for every tile and year given. Reads band metadata only.
Usage: check_map_names.py <models dir> <v4r root> <tiles file> <year>..."""
import pickle, sys
from pathlib import Path
from osgeo import gdal
sys.path.insert(0, str(Path(__file__).parent))
from map_predict import year_name
gdal.UseExceptions()
models, v4, tiles, years = Path(sys.argv[1]), Path(sys.argv[2]), open(sys.argv[3]).read().split(), [int(y) for y in sys.argv[4:]]
bad = 0
for y in years:
    names = list(pickle.load(open(models / ('V4R60_%d' % y) / 'pcttreeTopN.bin', 'rb')))
    for t in tiles:
        cache = {}
        for c in names:
            fam = c.split('-')[0]
            if fam not in cache:
                ds = gdal.Open(str(v4 / str(y) / t / str(y) / '3-Metrics' / ('%s.tif' % fam)))
                cache[fam] = {ds.GetRasterBand(i).GetMetadata().get('Description', '').replace(' ', '-') for i in range(1, ds.RasterCount + 1)}
            if year_name(c, y) not in cache[fam]:
                bad += 1; print('MISSING', y, t, c, '->', year_name(c, y))
    print(y, 'tiles', len(tiles), 'predictors', len(names), 'missing so far', bad)
print('RESULT', 'PASS' if bad == 0 else 'FAIL')
sys.exit(1 if bad else 0)
