"""Composites (model/CompositeDayFile.py, PGE61 compositing) and metrics (model/Metrics.py) on synthetic MOD44C inputs (6 x 6)."""
from pathlib import Path

import numpy as np
import pytest

import synthetic as syn

ROOT = Path(__file__).resolve().parents[1]
P1 = [(2019, 65), (2019, 81), (2019, 97)]          # the inputs of the first composite period (DOY 065)


def _value(cloudy=(), lowSun=(), b1=None):
    """Default inputs, with the days in cloudy flagged cloudy (state bits 0-1 = 1), the days in lowSun at solar zenith 75,
    and Band_1 of chosen days replaced (b1 = {(year, day): value})."""
    def f(y, d, band):
        a = syn.defaultValue(y, d, band)
        if band == 'state' and (y, d) in cloudy:
            a = a | 1
        if band == 'solar_zenith' and (y, d) in lowSun:
            a = np.full_like(a, 75)
        if band == 'Band_1' and b1 and (y, d) in b1:
            a = np.full_like(a, b1[(y, d)])
        return a
    return f


def _metrics(tmp_path, **kw):
    syn.writeInputs(tmp_path / 'in', 'h12v02', 2019, _value(**{k: v for k, v in kw.items() if k != 'skip'}),
                    skip=kw.get('skip', ()))
    return syn.metricsRoot(tmp_path / 'in', tmp_path / 'metrics')


def _period1(mt, band='Band_1'):
    cube, xref = mt.getBandCube(band)
    return cube[xref['2019065']]


def test_composite_is_mean_of_clear_days(smallTile, tmp_path):
    mt = _metrics(tmp_path)
    assert np.all(_period1(mt) == 1000 + (65 + 81 + 97) // 3)
    q, c = mt.getPeriodFlags()
    assert not q.any() and not c.any()


def test_cloudy_day_left_out(smallTile, tmp_path):
    mt = _metrics(tmp_path, cloudy=[(2019, 81)], b1={(2019, 81): 5000})
    assert np.all(_period1(mt) == (1065 + 1097) // 2)


def test_fallback_mean_and_flags(smallTile, tmp_path):
    mt = _metrics(tmp_path, cloudy=P1)
    assert np.all(_period1(mt) == 1000 + (65 + 81 + 97) // 3)       # no day passes: the mean of all days
    q, c = mt.getPeriodFlags()
    assert np.all(q == 1) and np.all(c == 1)                         # period 1 (bit 0) fell back, no clear day


def test_low_sun_day_left_out_and_substituted(smallTile, tmp_path):
    mt = _metrics(tmp_path, lowSun=[(2019, 65)], b1={(2019, 65): 3000})
    assert np.all(_period1(mt) == (1081 + 1097) // 2)               # solar zenith 75 fails QA (0 < solz < 65)
    mt2 = _metrics(tmp_path / 'b', lowSun=P1)                         # every day low sun: fallback of substituted values
    assert np.all(_period1(mt2) == 16000)


def test_missing_day(smallTile, tmp_path):
    mt = _metrics(tmp_path, skip=[(2019, 81)])
    assert np.all(_period1(mt) == (1065 + 1097) // 2)


def test_metrics_follow_the_period_values(smallTile, tmp_path):
    mt = _metrics(tmp_path, b1={(2019, 209): 9000, (2019, 225): 9000, (2019, 241): 9000})
    cube, xref = mt.getBandCube('Band_1')
    assert np.array_equal(mt.getMetricFromRf('BandReflMax-Band_1'), cube.max(0))
    assert np.array_equal(mt.getMetricFromRf('BandReflMin-Band_1'), cube.min(0))
    assert np.allclose(mt.getMetricFromRf('Lowest3MeanBandRefl-Band_1'), np.sort(cube, 0)[:3].mean(0), atol=1)
    assert np.array_equal(mt.getMetricFromRf('UnsortedMonthlyBands-Band_1-Day_2019209'), cube[xref['2019209']])
    assert (Path(mt._metricsDir) / 'BandReflMax.tif').exists()        # written once, then read back


@pytest.mark.parametrize('listName', ['predictors_tree.txt', 'predictors_bare.txt'])
def test_product_predictors_resolve(smallTile, tmp_path, listName):
    mt = _metrics(tmp_path)
    names = (ROOT / 'docs' / listName).read_text().split()
    for n in names:
        v = np.asarray(mt.getMetricFromRf(n))
        assert v.shape == (syn.SIZE, syn.SIZE) and np.isfinite(v).all(), n
