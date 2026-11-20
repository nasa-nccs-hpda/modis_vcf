"""v2.0.0 composites from dailies: Sensor, DailiesComposite (periods, days, per-day tests, options), AquaFill (merge rule)
and TileComposites (source record, Aqua availability, Aqua_Fill bits), on small synthetic arrays.  The full compositor is
proved on real dailies by the exactness gates (byte for byte against the Aqua-Terra Experiment runs)."""
import datetime
import json

import numpy as np
import pytest

from modis_vcf.model import AquaFill, DailiesComposite as DC, Sensor, TileComposites


# ---------------------------------------------------------------------------- Sensor
def test_sensors():
    assert Sensor.get('terra').products == ('MOD09GA', 'MOD09GQ', 'MODTBGA')
    assert Sensor.get('aqua').products == ('MYD09GA', 'MYD09GQ', 'MYDTBGA')
    assert 'Band_6' in Sensor.TERRA.validTest and 'Band_6' not in Sensor.AQUA.validTest
    assert 'Band_5' not in Sensor.TERRA.validTest and Sensor.AQUA.band6ValidDaysOnly and not Sensor.TERRA.band6ValidDaysOnly
    with pytest.raises(ValueError):
        Sensor.get('viirs')


# ---------------------------------------------------------------------------- periods and days
def test_periods_and_days():
    ps = DC.periods(2019)
    assert ps == ['2019065', '2019113', '2019161', '2019209', '2019257', '2019305', '2019353', '2020033']
    assert [len(DC.daysOf(p)) for p in ps] == [48, 48, 48, 48, 48, 48, 45, 32]
    assert DC.daysOf('2020033')[-1] == datetime.date(2020, 3, 4)            # DOY 064 of 2020, the end of year 2019
    assert DC.processingYear('2020033') == 2019 and DC.processingYear('2019353') == 2019
    # every day of the processing year exactly once, in order
    days = [d for p in ps for d in DC.daysOf(p)]
    assert days == [datetime.date(2019, 3, 6) + datetime.timedelta(k) for k in range(365)]
    # leap processing year 2019 -> 2020 above; 2023 -> 2024: 353 runs to 2024 DOY 032, 033 to 2024 DOY 064
    assert len(DC.daysOf('2023353')) == 45 and DC.daysOf('2024033')[-1] == datetime.date(2024, 3, 4)


def test_options():
    o = DC.CompositeOptions()
    assert (o.polarFill, o.otherOrbit, o.f2Rank, o.f3, o.f3Guard, o.rules) == (True, False, False, 'mean3', True, 'b1')
    with pytest.raises(ValueError):
        DC.CompositeOptions(f3='mean2')
    with pytest.raises(ValueError):
        DC.CompositeOptions(rules='b3')


def test_on_globe():
    g = DC.onGlobe('h17v00')
    assert g.shape == (DC.N, DC.N) and g[-1, DC.N // 2 - 1] and not g[0, 0]   # centre of the bottom row on, corner off
    assert DC.onGlobe('h18v08').all()


# ---------------------------------------------------------------------------- per-day tests
def _day(n=6, **kw):
    """One synthetic day of n pixels: clear, low aerosol, sun 30, VZ 10, valid bands."""
    d = {b: np.full(n, 500, np.int16) for b in DC.BANDS}
    d['Band_2'] = np.full(n, 3000, np.int16)
    d['Band_31'] = np.full(n, 29000, np.int16)
    d['state'] = np.full(n, 1 << 6, np.uint16)            # aerosol low (01), cloud 00
    d['solz'] = np.full(n, 3000, np.int16)
    d['vz'] = np.full(n, 1000, np.int16)
    d['same'] = np.ones(n, bool)
    for k, v in kw.items():
        d[k] = np.asarray(v, d[k].dtype if k in d else None)
    return d


def test_masks_good_and_sun_limit():
    m = DC.masks(_day(solz=[3000, 7150, 7200, 0, -32767, 6900]), Sensor.TERRA)
    assert m['good'].tolist() == [True, True, False, False, False, True]      # 0 < solz < 72
    assert m['lowsun'].tolist() == [False, False, True, True, False, False]   # clamp at >= 72 or <= 0 (present days)
    b0 = DC.masks(_day(solz=[3000, 6600, 7100, 0, 3000, 6900]), Sensor.TERRA, rules='b0')
    assert b0['good'].tolist() == [True, False, False, False, True, False]    # B0: solz < 65
    assert b0['lowsun'].tolist() == [False, False, True, True, False, False]  # B0: > 70


def test_masks_fallback_classes():
    st = [1 << 6, (1 << 6) | (1 << 13), (1 << 6) | (1 << 2), 3 << 6, (1 << 6) | 1, (1 << 6) | 1]
    m = DC.masks(_day(state=st, Band_2=[3000, 3000, 3000, 3000, 3000, 1000]), Sensor.TERRA)
    assert m['good'].tolist() == [True, False, False, False, False, False]
    assert m['f1'].tolist() == [False, True, False, False, False, False]      # adjacency only
    assert m['f2'].tolist() == [False, False, True, True, False, False]       # shadow / high aerosol
    assert m['f3'].tolist() == [False, False, False, False, True, False]      # cloudy; NIR < 0.15 fails the guard
    assert DC.masks(_day(state=st, Band_2=[3000] * 5 + [1000]), Sensor.TERRA, f3guard=False)['f3'][5]


def test_aqua_band6_not_tested():
    d = _day(Band_6=[-28672] * 6)
    assert not DC.masks(d, Sensor.TERRA)['good'].any() and DC.masks(d, Sensor.AQUA)['good'].all()


def test_pick1km():
    nobs = np.array([[1, 3]])                       # 1 km pixel 0: layer 1 only; pixel 1: 2 extra observations
    iobs = np.array([[0, 0, 1, 2], [0, 255, 2, 3]])   # 500 m: 2 x 4 over the 1 x 2 grid
    pk = DC.Pick1km(nobs, iobs)
    v = pk.take(np.array([[10, 20]], np.int16), np.array([21, 22], np.int16), -1)
    assert v.tolist() == [[10, 10, 21, 22], [10, -1, 22, -1]]        # pointer past the list (3 > 2 extra) = missing


# ---------------------------------------------------------------------------- AquaFill
def _period(root, tile, period, cls, b6n=None, value=0, qa=1):
    (root / '2-Composites').mkdir(parents=True, exist_ok=True)
    (root / 'diag').mkdir(parents=True, exist_ok=True)
    n = cls.shape[0]
    for b in DC.BANDS:
        np.full((n, n), value, np.int16).tofile(DC.compositeName(str(root), tile, period, b))
        for k in ('qa', 'cloudfb'):
            np.save(DC.compositeName(str(root), tile, period, b, '-%s.npy' % k), np.full((n, n), qa, np.uint8))
    np.save(DC.diagName(str(root), tile, period, 'fbclass'), cls.astype(np.int8))
    for k, dt in (('solz_used', np.float32), ('vz_used', np.float32), ('gooddays', np.int16), ('group', np.int8)):
        np.save(DC.diagName(str(root), tile, period, k), np.full((n, n), value, dt))
    if b6n is not None:
        np.save(DC.diagName(str(root), tile, period, 'b6_n'), b6n.astype(np.int16))


def test_aqua_fill_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(AquaFill, 'N', 2)
    t, p = 'h12v09', '2019161'
    _period(tmp_path / 'T', t, p, np.array([[0, 3], [5, 2]]), value=1)          # Terra: good, F3, none, F2
    _period(tmp_path / 'A', t, p, np.array([[0, 0], [0, 1]]), b6n=np.array([[2, 2], [0, 2]]), value=2, qa=0)
    s = AquaFill.merge(t, p, str(tmp_path / 'T'), str(tmp_path / 'A'), str(tmp_path / 'O'), diag=True)
    out = str(tmp_path / 'O')
    fbc = np.load(DC.diagName(out, t, p, 'fbclass'))
    assert fbc.tolist() == [[0, 7], [5, 2]]          # Terra good kept; F3 -> Aqua; no Aqua Band 6 -> Terra; Aqua F1 unused
    assert np.load(DC.diagName(out, t, p, 'fill_src')).tolist() == [[0, 1], [0, 0]]
    b1 = np.fromfile(DC.compositeName(out, t, p, 'Band_1'), np.int16).reshape(2, 2)
    assert b1.tolist() == [[1, 2], [1, 1]]
    assert np.load(DC.compositeName(out, t, p, 'Band_1', '-qa.npy')).tolist() == [[1, 0], [1, 1]]
    assert s['fill_share'] == 0.25 and DC.isDone(out, t, p)
    assert np.load(DC.diagName(out, t, p, 'solz_used')).tolist() == [[1, 2], [1, 1]]     # diag: Aqua's at class 7
    AquaFill.merge(t, p, str(tmp_path / 'T'), str(tmp_path / 'A'), str(tmp_path / 'O3'))   # diag off (default)
    import os
    assert sorted(os.listdir(tmp_path / 'O3' / 'diag')) == ['%s-%s-%s' % (t, p, k) for k in ('fbclass.npy', 'fill_src.npy', 'summary.json')]
    # no Aqua dailies in the period: Terra unchanged
    AquaFill.merge(t, p, str(tmp_path / 'T'), None, str(tmp_path / 'O2'))
    assert np.load(DC.diagName(str(tmp_path / 'O2'), t, p, 'fbclass')).tolist() == [[0, 3], [5, 2]]


# ---------------------------------------------------------------------------- TileComposites
def test_source_and_aqua_bits(tmp_path, monkeypatch):
    monkeypatch.setattr(DC, 'N', 2)
    root = tmp_path / 'h12v09' / '2019'
    assert TileComposites.source(root) is None and not TileComposites.isDailies(root)
    assert TileComposites.aquaFillBits(root, 'h12v09', 2019) is None
    root.mkdir(parents=True)
    (root / 'diag').mkdir()
    for n, p in enumerate(DC.periods(2019)):
        np.save(DC.diagName(str(root), 'h12v09', p, 'fill_src'), np.array([[n == 0, n == 7], [0, n in (1, 2)]], np.uint8))
    (root / TileComposites.SOURCE).write_text(json.dumps({'input': 'dailies', 'product': 'MCD44B', 'aquaFill': True}))
    assert TileComposites.isDailies(root)
    assert TileComposites.aquaFillBits(root, 'h12v09', 2019).tolist() == [[1, 128], [0, 6]]


def test_aqua_available(tmp_path):
    d = tmp_path / 'MYD09GA' / '2020'
    d.mkdir(parents=True)
    assert not TileComposites.aquaAvailable(str(tmp_path), 2019)
    (d / 'MYD09GA.A2020070.h12v09.007.x.h5').write_bytes(b'')                # after the processing year 2019
    assert not TileComposites.aquaAvailable(str(tmp_path), 2019)
    (d / 'MYD09GA.A2020064.h12v09.007.x.h5').write_bytes(b'')                # its last day
    assert TileComposites.aquaAvailable(str(tmp_path), 2019)


def test_build_refuses_other_settings(tmp_path):
    root = tmp_path / 'h12v09' / '2019'
    root.mkdir(parents=True)
    (root / TileComposites.SOURCE).write_text(json.dumps({'input': 'dailies', 'product': 'MOD44B', 'sensors': ['terra'],
                                                          'aquaFill': False, 'options': {}}))
    with pytest.raises(RuntimeError, match='new metrics root'):
        TileComposites.buildTile('h12v09', 2019, str(tmp_path), root, tmp_path / 'w', aquaFill=True)


def test_map_product_from_composites(tmp_path):
    """VcfPredict names the product from the tile's composites.json; an explicit product must agree."""
    from modis_vcf.model.VcfPredict import VcfPredict
    v = VcfPredict.__new__(VcfPredict)
    v._metricsDir = tmp_path
    v._product = None
    assert v._tileProduct('h12v09', 2019) == 'MOD44B'                       # MOD44C root
    root = tmp_path / 'h12v09' / '2019'
    root.mkdir(parents=True)
    (root / TileComposites.SOURCE).write_text(json.dumps({'input': 'dailies', 'product': 'MCD44B', 'aquaFill': True}))
    assert v._tileProduct('h12v09', 2019) == 'MCD44B'
    v._product = 'MOD44B'
    with pytest.raises(ValueError, match='composites make MCD44B'):
        v._tileProduct('h12v09', 2019)
