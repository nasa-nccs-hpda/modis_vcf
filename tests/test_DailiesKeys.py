"""v2.0.2 keys mode: DailiesComposite.composite(keys=...) equals full mode at the keys (synthetic days on a small tile), with
fill elsewhere; TileComposites.keysFromManifest; the settings that keys mode refuses."""
import json

import numpy as np
import pandas as pd
import pytest

from modis_vcf.model import DailiesComposite as DC
from modis_vcf.model import TileComposites as TC

SMALL = 8
TILE, PERIOD = 'h20v06', '2019161'


def _fakeReadDay(root, tile, day, sensor, keys=None):
    """A synthetic day on a SMALL x SMALL tile: mixed clear / cloudy / shadow / snow / low-sun / missing pixels, every 7th day
    missing; cut to keys like readDay."""
    n = DC.N
    if day.toordinal() % 7 == 0:
        return None, {p: None for p in sensor.products}
    r = np.random.default_rng(day.toordinal())
    d = {b: r.integers(-50, 9000, (n, n)).astype(np.int16) for b in DC.BANDS}
    d['Band_1'] = np.where(r.random((n, n)) < 0.05, DC.FILL, d['Band_1']).astype(np.int16)
    d['Band_31'] = r.integers(1, 32767, (n, n)).astype(np.int16)
    bits = r.choice([0, 1, 2, 4, 1 << 10, 1 << 12, 1 << 15, 64, 128, 192, 64 | (1 << 15)], (n, n))
    bits[0] = r.choice([1, 2, 1 | 64], n)                             # row 0 always cloudy (F3 / F4 fallback)
    bits[1] = r.choice([4, 192, 64 | 4], n)                           # row 1 never clear: shadow / high aerosol / adjacent (F1, F2)
    d['state'] = np.where(r.random((n, n)) < 0.05, 65535, bits).astype(np.uint16)
    d['state'][2, :2] = 65535                                         # two pixels never present (class 5)
    d['solz'] = r.choice([-500, 2000, 4000, 6000, 7100, 7300, 8000], (n, n)).astype(np.int16)
    d['vz'] = r.integers(0, 6500, (n, n)).astype(np.int16)
    d['b31_over'] = np.zeros((n, n), bool)
    d['layer1_1km'] = r.random((n, n)) < 0.8
    d['qc500'] = np.zeros((n, n), np.uint32)
    d['same'] = r.random((n, n)) < 0.95
    if keys is not None:
        d = {k: v.ravel()[keys] for k, v in d.items()}
    return d, {p: '%s.%s' % (p, day) for p in sensor.products}


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(DC, 'N', SMALL)
    monkeypatch.setattr(DC, 'M', SMALL // 2)
    monkeypatch.setattr(DC, 'readDay', _fakeReadDay)
    return SMALL


def _bin(out, band):
    return np.fromfile(DC.compositeName(str(out), TILE, PERIOD, band), np.int16).reshape(SMALL, SMALL)


@pytest.mark.parametrize('opts', [{}, {'polarFill': False}, {'f3': 'single'}, {'rules': 'b0'}])
def test_keys_equal_full_mode(small, tmp_path, opts):
    o = DC.CompositeOptions(**opts)
    full = DC.composite(TILE, PERIOD, 'x', str(tmp_path / 'full'), options=o)
    keys = np.array(sorted(set(np.random.default_rng(1).choice(SMALL * SMALL, 20, replace=False)) | {0, 3, 9, 12, 16, 17}), np.int64)
    kp = DC.composite(TILE, PERIOD, 'x', str(tmp_path / 'keys'), options=o, keys=keys)
    off = np.ones(SMALL * SMALL, bool)
    off[keys] = False
    for b in DC.BANDS:
        a, k = _bin(tmp_path / 'full', b).ravel(), _bin(tmp_path / 'keys', b).ravel()
        assert np.array_equal(a[keys], k[keys]), b
        assert (k[off] == DC.FILL).all()
        for sfx in ('-qa.npy', '-cloudfb.npy'):
            qa = np.load(DC.compositeName(str(tmp_path / 'full'), TILE, PERIOD, b, sfx)).ravel()
            qk = np.load(DC.compositeName(str(tmp_path / 'keys'), TILE, PERIOD, b, sfx)).ravel()
            assert np.array_equal(qa[keys], qk[keys]) and not qk[off].any()
    fa = np.load(DC.diagName(str(tmp_path / 'full'), TILE, PERIOD, 'fbclass')).ravel()
    fk = np.load(DC.diagName(str(tmp_path / 'keys'), TILE, PERIOD, 'fbclass')).ravel()
    assert np.array_equal(fa[keys], fk[keys]) and (fk[off] == -1).all()
    assert len(set(fa[keys])) >= 3                                   # the keys see good and fallback classes
    assert kp['keys'] == len(keys) and full['keys'] is None
    assert json.loads(open(DC.diagName(str(tmp_path / 'keys'), TILE, PERIOD, 'summary.json')).read())['keys'] == len(keys)


def test_keys_refused_settings(small, tmp_path):
    keys = np.array([1, 5, 9], np.int64)
    with pytest.raises(ValueError, match='diag'):
        DC.composite(TILE, PERIOD, 'x', str(tmp_path), keys=keys, diag=True)
    with pytest.raises(ValueError, match='otherOrbit'):
        DC.composite(TILE, PERIOD, 'x', str(tmp_path), keys=keys, options=DC.CompositeOptions(otherOrbit=True))
    for bad in ([5, 1], [1, 1], [-1, 2], [SMALL * SMALL]):
        with pytest.raises(ValueError, match='sorted'):
            DC.composite(TILE, PERIOD, 'x', str(tmp_path), keys=np.array(bad, np.int64))


def test_keys_from_manifest(tmp_path):
    p = tmp_path / 'm.parq'
    pd.DataFrame({'part': ['Train', 'Test', 'Train', 'Train'], 'row': range(4),
                  'tid-year': ['h20v06-2019', 'h20v06-2019', 'h16v02-2019', 'h20v06-2019'],
                  'x': [2, 0, 7, 2], 'y': [3, 9, 7, 3], 'label': [1, 2, 3, 4]}).to_parquet(p)
    k = TC.keysFromManifest(p, 'h20v06')
    assert list(k) == [9, 2 * 4800 + 3]                               # x is the row: flat = x * 4800 + y, sorted, unique
    i = TC.keysInfo(p, k)
    assert i['n'] == 2 and len(i['sha256']) == 64
    assert len(TC.keysFromManifest(p, 'h09v05')) == 0


def test_build_tile_keys_refuses_aqua_and_diag(tmp_path):
    p = tmp_path / 'm.parq'
    pd.DataFrame({'part': ['Train'], 'row': [0], 'tid-year': ['h20v06-2019'], 'x': [1], 'y': [1], 'label': [1]}).to_parquet(p)
    with pytest.raises(ValueError, match='Terra only'):
        TC.buildTile('h20v06', 2019, str(tmp_path), tmp_path / 'r', tmp_path / 'w', aquaFill=True, keysManifest=p)
    with pytest.raises(ValueError, match='Terra only'):
        TC.buildTile('h20v06', 2019, str(tmp_path), tmp_path / 'r', diag=True, keysManifest=p)
    with pytest.raises(ValueError, match='no label pixels'):
        TC.buildTile('h09v05', 2019, str(tmp_path), tmp_path / 'r', keysManifest=p)


def test_year_run_compositeKeys_rules(tmp_path, monkeypatch):
    from test_YearRun import spec as _spec, _yr                       # noqa: F401
    d, s = _spec.__wrapped__(tmp_path, monkeypatch) if hasattr(_spec, '__wrapped__') else _specFixture(tmp_path, monkeypatch)
    s.pop('cloudFallbackTree')
    s['input'], s['dailiesDir'] = 'dailies', str(d)
    s['compositeKeys'] = str(d / 'nope.parq')
    with pytest.raises(ValueError, match='existing training manifest'):
        _yr(d, s, 'k1.json')
    m = d / 'keys.parq'
    pd.DataFrame({'part': ['Train'], 'row': [0], 'tid-year': ['h12v02-2019'], 'x': [1], 'y': [1], 'label': [1]}).to_parquet(m)
    s['compositeKeys'] = str(m)
    with pytest.raises(ValueError, match='no mapTiles'):
        _yr(d, s, 'k2.json')
    s['mapTiles'] = []
    with pytest.raises(ValueError, match='splitOnly'):
        _yr(d, s, 'k3.json')
    s['models']['tree']['splitOnly'] = True
    y = _yr(d, s, 'k4.json')
    assert y.compositeKeys == str(m) and sorted(y.tasks) == ['metrics:h12v02', 'split:tree']
    s['compositeDiag'] = True
    with pytest.raises(ValueError, match='without aquaFill and without compositeDiag'):
        _yr(d, s, 'k5.json')


def _specFixture(tmp_path, monkeypatch):
    import test_YearRun as T
    return T.spec.__pytest_wrapped__.obj(tmp_path, monkeypatch)
