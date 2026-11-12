"""model/CloudFallback.py: cloud-fallback composites and counts on synthetic inputs (the period pool run in threads, so the
small tile applies)."""
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np

import synthetic as syn
from modis_vcf.model import CloudFallback

P1 = [(2019, 65), (2019, 81), (2019, 97)]
RED = {(2019, 65): 500, (2019, 81): 300, (2019, 97): 400}


def _value(y, d, band):
    a = syn.defaultValue(y, d, band)
    if (y, d) in RED:
        if band == 'state':
            a = a | 1                                            # every input of period 1 cloudy: the period falls back
            if (y, d) == (2019, 81):
                a[0, 0] |= 4                                     # ... and the lowest-red input is shadow at pixel (0, 0)
        if band == 'Band_1':
            a = np.full_like(a, RED[(y, d)])
        if band == 'Band_2':
            a = np.full_like(a, 2000)                            # NIR >= 1500, NDVI >= 0
    return a


def test_build_tile(smallTile, tmp_path, monkeypatch):
    syn.writeInputs(tmp_path / 'in', 'h12v02', 2019, _value)
    std = syn.metricsRoot(tmp_path / 'in', tmp_path / 'std')
    std.getBandCube('Band_1')                                    # the standard composites and their QA flags
    import modis_vcf.model.VcfPredict as vp
    monkeypatch.setattr(vp, 'MOD44_DIR', tmp_path / 'in')
    monkeypatch.setattr(CloudFallback, 'ProcessPoolExecutor', lambda n, mp_context=None: ThreadPoolExecutor(n))
    s = CloudFallback.buildTile('h12v02', 2019, tmp_path / 'std', tmp_path / 'cf', tmp_path / 'counts', workers=2)
    assert s['pick_band1_mismatch'] == 0 and s['alignment_min_identical'] >= 0.999
    comp = np.fromfile(tmp_path / 'cf' / 'h12v02' / '2019' / '2-Composites' / 'MOD44-h12v02-2019065-Band_1.bin',
                       np.int16).reshape(syn.SIZE, syn.SIZE)
    expected = np.full((syn.SIZE, syn.SIZE), 300); expected[0, 0] = 400           # the lowest red; not the shadow input
    assert np.array_equal(comp, expected)
    other = np.fromfile(tmp_path / 'cf' / 'h12v02' / '2019' / '2-Composites' / 'MOD44-h12v02-2019113-Band_1.bin', np.int16)
    assert np.all(other == 1000 + (113 + 129 + 145) // 3)                          # clear period: the standard composite
    used = np.load(tmp_path / 'counts' / 'h12v02_cloudfallback_used_count.npy')
    counts = np.load(tmp_path / 'counts' / 'h12v02_counts_snow.npz')
    assert np.all(used == 1) and np.all(counts['CUR_Z'] == 1) and np.all(counts['CUR_ZS'] <= 1)
    assert json.loads((tmp_path / 'counts' / 'h12v02_cloudfallback_build.json').read_text())['summary']['tile'] == 'h12v02'
