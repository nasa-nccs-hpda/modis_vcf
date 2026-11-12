"""model/TrainingManifest.py: sample (labels -> manifest), build (manifest -> split, from metrics) and the acceptance test of the
cloud-fallback tree split, on synthetic labels and inputs."""
import json

import numpy as np
import pandas as pd
import pytest

import synthetic as syn
from modis_vcf.model import TrainingManifest as TM
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44

PRED = ['BandReflMax-Band_1', 'UnsortedMonthlyBands-NDVI-Day_2019065', 'CUnsMean16-Band_6']


def _labels(tmp_path, n=36):
    a = np.full((4800, 4800), 255, np.uint8)
    a[:syn.SIZE, :syn.SIZE] = np.arange(syn.SIZE * syn.SIZE).reshape(syn.SIZE, syn.SIZE) % 101   # labels inside the small tile
    f = tmp_path / 'labels_h12v02.out'
    a.tofile(f)
    return {'h12v02': f}


def test_sample_deterministic(tmp_path):
    lab = _labels(tmp_path)
    m1 = TM.sample(lab, 'pctbare', 2019, tmp_path / 'a.parq', 'test')
    m2 = TM.sample(lab, 'pctbare', 2019, tmp_path / 'b.parq', 'test')
    a, meta = TM.read(tmp_path / 'a.parq')
    assert pd.read_parquet(tmp_path / 'b.parq').equals(a) and m1['sha256'] == m2['sha256']
    assert len(a) == 36 and set(a.part) <= {'Train', 'Test'} and meta['trainingType'] == 'pctbare'
    assert a.x.max() < syn.SIZE and a.y.max() < syn.SIZE and a.label.max() <= 100      # no-data 255 left out
    with pytest.raises(RuntimeError, match='exists'):
        TM.sample(lab, 'pctbare', 2019, tmp_path / 'a.parq', 'test')


def test_sample_refuses_wrong_size(tmp_path):
    np.zeros(100, np.uint8).tofile(tmp_path / 'bad.out')
    with pytest.raises(RuntimeError, match='4800'):
        TM.sample({'h12v02': tmp_path / 'bad.out'}, 'pctbare', 2019, tmp_path / 'm.parq', 'test')


@pytest.fixture(scope='module')
def built(smallTileModule, tmp_path_factory):
    d = tmp_path_factory.mktemp('tm')
    syn.writeInputs(d / 'in', 'h12v02', 2019)
    mt = syn.metricsRoot(d / 'in', d / 'metrics')
    for n in PRED:                                               # the metrics task first, as in a year run
        mt.getMetricFromRf(n)
    mt = syn.metricsRoot(d / 'in', d / 'metrics')                # a fresh reader: values as written (int16 files)
    TM.sample(_labels(d), 'pctbare', 2019, d / 'm.parq', 'test')
    pt = ProductTypeMod44(d / 'in')
    for k in ('s1', 's2'):
        TM.build(d / 'm.parq', PRED, lambda t: d / 'metrics', 2019, pt, d / k)
    return d, mt


def test_build_values_and_determinism(built):
    d, mt = built
    k = pd.read_parquet(d / 's1' / 'keysTrain.parq')
    x = pd.read_parquet(d / 's1' / 'xTrain.parq')
    assert list(x.columns) == PRED and len(x) == len(k) > 0
    for n in PRED:
        assert np.array_equal(x[n].to_numpy(), np.asarray(mt.getMetricFromRf(n))[k.x.to_numpy(), k.y.to_numpy()])
    for f in ('xTrain.parq', 'yTrain.parq', 'keysTrain.parq', 'xTest.parq'):
        assert pd.read_parquet(d / 's1' / f).equals(pd.read_parquet(d / 's2' / f))


def test_accept(built):
    d, _ = built
    counts = d / 'counts'
    counts.mkdir(exist_ok=True)
    np.save(counts / 'h12v02_cloudfallback_used_count.npy', np.zeros((syn.SIZE, syn.SIZE), np.uint8))
    rep = TM.accept(d / 's2', d / 's1', counts, lambda t: d / 'metrics', 2019, d / 'accept.json')
    assert rep['RESULT'] == 'PASS' and json.loads((d / 'accept.json').read_text())['RESULT'] == 'PASS'
    x = pd.read_parquet(d / 's2' / 'xTrain.parq')
    x.iloc[0, 0] += 1                                              # a row the fallback never replaced now differs
    x.to_parquet(d / 's2' / 'xTrain.parq')
    assert TM.accept(d / 's2', d / 's1', counts, lambda t: d / 'metrics', 2019, d / 'accept2.json')['RESULT'] == 'FAIL'
