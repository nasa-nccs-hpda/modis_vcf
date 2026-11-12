"""model/PooledSplit.py: the pooled bare split from per-year splits (synthetic splits of 3 years)."""
import json

import numpy as np
import pandas as pd
import pytest

from modis_vcf.model import PooledSplit

PRED = ['p0', 'p1', 'p2']
YEARS = (2000, 2001, 2002)


def _splits(tmp_path, n=3000, seed=5):
    rng = np.random.default_rng(seed)
    tiles = rng.choice(['h12v02', 'h21v07'], n)
    x, y, lab = rng.integers(0, 4800, n), rng.integers(0, 4800, n), rng.integers(0, 101, n)
    out = {}
    for yr in YEARS:
        d = tmp_path / ('split%d' % yr)
        d.mkdir()
        pd.DataFrame({'tid-year': [t + '-%d' % yr for t in tiles], 'x': x, 'y': y}).to_parquet(d / 'keysTrain.parq', index=False)
        pd.DataFrame({'label': lab}).to_parquet(d / 'yTrain.parq', index=False)
        # Every predictor of year yr holds yr * 10 + its column number, so a row shows which year it came from.
        pd.DataFrame({p: np.full(n, yr * 10 + i, np.float32) for i, p in enumerate(PRED)}).to_parquet(d / 'xTrain.parq', index=False)
        out[yr] = d
    return out


def test_build_and_check(tmp_path):
    splits = _splits(tmp_path)
    m = PooledSplit.build(splits, PRED, tmp_path / 'pool', seed=0)
    x = pd.read_parquet(tmp_path / 'pool' / 'xTrain.parq')
    a = pd.read_parquet(tmp_path / 'pool' / 'assign.parq')
    assert m['rows'] == 3000 and sorted(m['rowsPerYear']) == list(YEARS)
    assert max(m['rowsPerYear'].values()) - min(m['rowsPerYear'].values()) <= len(YEARS)
    assert (x['p0'] // 10 == a['year']).all() and (x['p2'] - x['p0'] == 2).all()      # whole rows from one year
    assert pd.read_parquet(tmp_path / 'pool' / 'yTrain.parq').equals(pd.read_parquet(splits[2000] / 'yTrain.parq'))
    summary = PooledSplit.check(tmp_path / 'pool', splits)
    assert summary['maxStratumSpread'] <= 1 and not summary['yearsWithDifferences']
    assert json.loads((tmp_path / 'pool' / 'pooled_split.json').read_text())['seed'] == 0


def test_deterministic(tmp_path):
    splits = _splits(tmp_path)
    PooledSplit.build(splits, PRED, tmp_path / 'a', seed=0)
    PooledSplit.build(splits, PRED, tmp_path / 'b', seed=0)
    PooledSplit.build(splits, PRED, tmp_path / 'c', seed=1)
    a, b, c = (pd.read_parquet(tmp_path / k / 'assign.parq')['year'] for k in 'abc')
    assert a.equals(b) and not a.equals(c)
    for f in PooledSplit.FILES:
        assert (tmp_path / 'a' / f).read_bytes() == (tmp_path / 'b' / f).read_bytes()


def test_refuses_differing_keys_and_labels(tmp_path):
    splits = _splits(tmp_path)
    k = pd.read_parquet(splits[2001] / 'keysTrain.parq')
    k.loc[0, 'x'] += 1
    k.to_parquet(splits[2001] / 'keysTrain.parq', index=False)
    with pytest.raises(ValueError, match='keys of 2001'):
        PooledSplit.build(splits, PRED, tmp_path / 'pool')
    (tmp_path / 'b').mkdir()
    splits = _splits(tmp_path / 'b')
    y = pd.read_parquet(splits[2002] / 'yTrain.parq')
    y.loc[0, 'label'] = (y.loc[0, 'label'] + 1) % 101
    y.to_parquet(splits[2002] / 'yTrain.parq', index=False)
    with pytest.raises(ValueError, match='labels of 2002'):
        PooledSplit.build(splits, PRED, tmp_path / 'pool2')


def test_check_detects_changed_output(tmp_path):
    splits = _splits(tmp_path)
    PooledSplit.build(splits, PRED, tmp_path / 'pool')
    x = pd.read_parquet(tmp_path / 'pool' / 'xTrain.parq')
    x.loc[0, 'p0'] = -1
    x.to_parquet(tmp_path / 'pool' / 'xTrain.parq', index=False)
    with pytest.raises(ValueError, match='SHA-256'):
        PooledSplit.check(tmp_path / 'pool', splits)


def test_refuses_existing_output(tmp_path):
    splits = _splits(tmp_path)
    (tmp_path / 'pool').mkdir()
    with pytest.raises(FileExistsError):
        PooledSplit.build(splits, PRED, tmp_path / 'pool')
