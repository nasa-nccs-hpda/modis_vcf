"""model/VcfPredict.py: the order of the map rules on a tile (predictions mocked): cloud-fallback tree blend, sum cap, v00 rule,
water / outside-projection codes, bare-priority rule; and a rule tile from MOD44W alone."""
import json
import logging
from unittest import mock

import numpy as np

from modis_vcf.model import RuleTile
from modis_vcf.model.VcfPredict import VcfPredict

W, O = VcfPredict.MASK_WATER, VcfPredict.MASK_OUT_OF_PROJ


def _vp(tmp_path, tid, mask, fallback=False, barePriority=True):
    vp = VcfPredict.__new__(VcfPredict)
    vp._logger = logging.getLogger('test')
    vp._waterMasks = {tid + '2019': mask}
    vp._treeCoverRf = vp._nonvegRf = object()
    vp._treeCoverTopN = vp._nonvegTopN = None
    vp._cloudFallbackTreeRf = object() if fallback else None
    vp._cloudFallbackTreeTopN = vp._cloudFallbackMetricsDir = None
    vp._cloudFallbackTreeAgg = 'median'
    vp._countsDir = tmp_path
    vp._barePriority = barePriority
    vp._treeCoverAgg = vp._nonvegAgg = 'median'
    vp._outDir = tmp_path
    vp._productType = vp._metricsDir = None
    vp._setMapFormat('geotiff')
    return vp


def _run(vp, tid, predictions, counts=None):
    written = {}
    flags = mock.Mock()
    flags.return_value.getPeriodFlags.return_value = (np.ones((1, 6), np.uint8), np.zeros((1, 6), np.uint8))
    preds = [(vp._maskPrediction(tid, 2019, np.array([p], np.int16)), np.zeros((1, 6), np.int16)) for p in predictions]
    with mock.patch.object(VcfPredict, '_runPrediction', side_effect=preds), \
         mock.patch.object(VcfPredict, '_write', side_effect=lambda t, y, n, a, *x, **k: written.__setitem__(n, np.asarray(a))), \
         mock.patch('modis_vcf.model.VcfPredict.CloudFallbackBlend.loadCounts', return_value=counts), \
         mock.patch('modis_vcf.model.VcfPredict.Metrics', flags):
        vp.runTileForYear(tid, 2019)
    return {k: v.tolist()[0] for k, v in written.items()}


def test_sum_cap_bare_priority_and_codes(tmp_path):
    mask = np.array([[0, 0, 0, 0, W, O]])
    # tree, raw bare: (60, 70) capped to 40; (5, 100) bare-priority -> 0 / 100; (5, 99) kept; (0, 30)
    w = _run(_vp(tmp_path, 'h12v02', mask), 'h12v02', [[60, 5, 5, 0, 0, 0], [70, 100, 99, 30, 0, 0]])
    assert w['Percent_Tree_Cover'] == [60, 0, 5, 0, 200, 250]
    assert w['Percent_NonVegetated'] == [40, 100, 95, 30, 200, 250]
    assert w['Percent_NonTree_Vegetation'] == [0, 0, 0, 70, 200, 250]
    assert w['Quality'] == [1, 1, 1, 1, 0, 0]                       # flags 0 on water / outside
    assert json.loads((tmp_path / 'h12v02-2019-chain.json').read_text())['barePriority'] == 1


def test_bare_priority_off(tmp_path):
    w = _run(_vp(tmp_path, 'h12v02', np.zeros((1, 6), int), barePriority=False), 'h12v02', [[5] * 6, [100] * 6])
    assert w['Percent_Tree_Cover'] == [5] * 6 and w['Percent_NonVegetated'] == [95] * 6


def test_blend_before_sum_cap(tmp_path):
    mask = np.zeros((1, 6), int)
    counts = np.array([[8, 8, 0, 8, 8, 8]], np.int16)
    # clear tree 60, fallback 90, w = 1 where counts 8 (clear 60 >= 50): tree 90; bare 30 capped to 10 after the blend
    w = _run(_vp(tmp_path, 'h21v07', mask, fallback=True), 'h21v07', [[60] * 6, [30] * 6, [90] * 6], counts)
    assert w['Percent_Tree_Cover'] == [90, 90, 60, 90, 90, 90]
    assert w['Percent_NonVegetated'] == [10, 10, 30, 10, 10, 10]
    chain = json.loads((tmp_path / 'h21v07-2019-chain.json').read_text())
    assert chain['blend'] and chain['tree_changed'] == 5


def test_no_blend_outside_rows(tmp_path):
    w = _run(_vp(tmp_path, 'h12v02', np.zeros((1, 6), int), fallback=True), 'h12v02', [[60] * 6, [30] * 6])
    assert w['Percent_Tree_Cover'] == [60] * 6                      # row v02: the fallback model is not even predicted


def test_rule_tile(tmp_path):
    vp = VcfPredict.forRuleTiles(tmp_path, mod44wDir=tmp_path, mapFormat='geotiff')
    vp._waterMasks = {'h16v16' + '2019': np.array([[0, W, O, 0]])}
    written = {}
    with mock.patch.object(VcfPredict, '_write', side_effect=lambda t, y, n, a, *x, **k: written.__setitem__(n, np.asarray(a))):
        vp.runTileForYear('h16v16', 2019)
    assert written['Percent_Tree_Cover'].tolist() == [[0, 200, 250, 0]]
    assert written['Percent_NonVegetated'].tolist() == [[100, 200, 250, 100]]
    assert written['Quality'].tolist() == [[RuleTile.ALL_PERIODS, 0, 0, RuleTile.ALL_PERIODS]]
    assert written['Percent_Tree_Cover_SD'].tolist() == [[0, 0, 0, 0]]
