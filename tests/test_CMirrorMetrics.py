"""C-mirror metrics (model/CMirrorMetrics.py): the C code's derived predictors (CUns*, CGreen*, CTherm*, CSortedRank*), which are
most of the product's non-period predictors.

The vectorised Python is checked against a literal per-pixel copy of the C code's loops (MOD_PR44B1/make_metrics.c: sums in int,
division truncated toward zero, the exchange sort that swaps on a strict '<' only), on synthetic values with many ties.

pytest tests/test_CMirrorMetrics.py
"""
from pathlib import Path

import numpy as np
import pytest

import synthetic as syn
from modis_vcf.model import CAttributes
from modis_vcf.model import CMirrorMetrics as CM

ROOT = Path(__file__).resolve().parents[1]
NB, NP = len(CAttributes.C_BANDS), 8                 # 9 bands (Band_1 .. Band_7, NDVI, Band_31) x 8 periods
NDVI, B31 = CM.NDVI_BAND, CM.THERMAL_BAND


# ----------------------------------------------------------------------------
# The C code, one pixel at a time (plain Python ints)
# ----------------------------------------------------------------------------
def cDiv(a: int, b: int) -> int:
    """C integer division (C99): the quotient truncated toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def cStats(p: list) -> dict:
    """The six statistics of one band's 8 period values, in the given period order."""
    return {'Mean13': cDiv(p[0] + p[1], 2),
            'Mean16': cDiv(p[0] + p[1] + p[2] + p[3], 4),
            'Mean19': cDiv(p[0] + p[1] + p[2] + p[3] + p[4] + p[5], 6),
            'Mean46': cDiv(p[2] + p[3], 2),
            'Mean79': cDiv(p[4] + p[5], 2),
            'Amp15': cDiv(p[0] + p[1] - p[2] - p[3], 2)}


def cExchangeSort(s: list, key: int) -> list:
    """make_metrics.c: for n < nn, swap whole periods (all bands) when key[n] < key[nn]."""
    s = [row[:] for row in s]
    for n in range(7):
        for nn in range(n + 1, 8):
            if s[key][n] < s[key][nn]:
                for k in range(NB):
                    s[k][n], s[k][nn] = s[k][nn], s[k][n]
    return s


def cCompute(pixel: list) -> dict:
    """{family: [value per band]} for one pixel; pixel is 9 lists of 8 ints."""
    out = {}
    for k in range(NB):
        for stat, v in cStats(pixel[k]).items():
            out.setdefault('CUns' + stat, []).append(v)
        for rank, v in enumerate(sorted(pixel[k], reverse=True)):
            out.setdefault('CSortedRank' + str(rank), []).append(v)
    for prefix, key in (('CGreen', NDVI), ('CTherm', B31)):
        s = cExchangeSort(pixel, key)
        for k in range(NB):
            out.setdefault(prefix + 'Max', []).append(s[k][0])
            for stat, v in cStats(s[k]).items():
                out.setdefault(prefix + stat, []).append(v)
    return out


def _cube(n: int = 60, seed: int = 0) -> np.ndarray:
    """(9, 8, n) int16 period values: reflectance-like bands, NDVI -1000..1000, Band_31 thermal; the first half of the pixels have
    NDVI and Band_31 drawn from 3 values only, so the sorts meet many ties."""
    rng = np.random.default_rng(seed)
    c = rng.integers(-100, 10000, (NB, NP, n))
    c[NDVI] = rng.integers(-1000, 1001, (NP, n))
    c[B31] = rng.integers(25000, 32000, (NP, n))
    h = n // 2
    c[NDVI, :, :h] = rng.choice([-50, 300, 700], (NP, h))
    c[B31, :, :h] = rng.choice([28000, 29000, 30000], (NP, h))
    return c.astype(np.int16)


# ----------------------------------------------------------------------------
# Unit tests of the functions
# ----------------------------------------------------------------------------
def test_truncDiv_truncates_toward_zero():
    x = np.arange(-25, 26)
    for d in (2, 4, 6):
        assert CM.truncDiv(x, d).tolist() == [cDiv(int(v), d) for v in x]
    assert CM.truncDiv(np.array([-3]), 2)[0] == -1      # numpy floor division would give -2


def test_compute_matches_the_c_loops():
    cube = _cube()
    got = CM.compute(cube)
    for i in range(cube.shape[2]):
        want = cCompute(cube[:, :, i].astype(int).tolist())
        for family, values in want.items():
            assert got[family][:, i].tolist() == values, (family, i)


def test_families_and_types():
    got = CM.compute(_cube(8))
    expected = {'CUns' + s for s in CAttributes.STATS} | {'CGreen' + s for s in CAttributes.STATS} | \
               {'CTherm' + s for s in CAttributes.STATS} | {'CGreenMax', 'CThermMax'} | {'CSortedRank%d' % r for r in range(8)}
    assert set(got) == expected
    for family, v in got.items():
        assert v.dtype == np.int16 and v.shape == (NB, 8), family


def test_ordering_high_to_low():
    cube = _cube()
    got = CM.compute(cube)
    assert np.array_equal(got['CGreenMax'][NDVI], cube[NDVI].max(0))
    assert np.array_equal(got['CThermMax'][B31], cube[B31].max(0))
    assert np.array_equal(got['CSortedRank0'], cube.max(1))
    assert np.array_equal(got['CSortedRank7'], cube.min(1))


def test_ties_are_never_swapped():
    """Equal NDVI in every period: the periods keep their order, so the CGreen statistics equal the unsorted ones."""
    cube = _cube(10)
    cube[NDVI] = 250
    got = CM.compute(cube)
    for s in CAttributes.STATS:
        assert np.array_equal(got['CGreen' + s], got['CUns' + s]), s
    assert np.array_equal(got['CGreenMax'], cube[:, 0])


def test_sums_do_not_overflow_int16():
    """Band_31 sums exceed int16 (6 x 30000); C sums in int, and so must the mirror."""
    cube = np.full((NB, NP, 1), 30000, np.int16)
    got = CM.compute(cube)
    assert got['CUnsMean19'][B31, 0] == 30000 and got['CThermMean16'][B31, 0] == 30000


# ----------------------------------------------------------------------------
# Through Metrics, as prediction reads them (synthetic MOD44C inputs)
# ----------------------------------------------------------------------------
def _randomInputs(y: int, d: int, band: str) -> np.ndarray:
    """Clear, QC-good days with per-pixel random band values (a fixed seed per day and band)."""
    if band in ('state', 'solar_zenith'):
        return syn.defaultValue(y, d, band)
    rng = np.random.default_rng(y * 1000 + d + 7919 * (syn.CQ_BANDS + syn.CH_BANDS).index(band))
    lo, hi = (25000, 32000) if band == 'Band_31' else (100, 6000)
    return rng.integers(lo, hi, (syn.SIZE, syn.SIZE)).astype(np.int16)


def _unsortedFromFile(mt) -> np.ndarray:
    """The written unsorted period values (int16, as computeWanted reads them), as (9, 8, rows, cols)."""
    name = 'metric' + CAttributes.UNSORTED
    mt.getMetric(name)
    values = {tuple(m.name.split('-')[1:]): m.value for m in mt.readMetric(mt._getOutName(name))}
    days = sorted({d for _, d in values})
    assert len(days) == 8
    return np.stack([np.stack([values[(b, d)] for d in days]) for b in CAttributes.C_BANDS]).astype(int)


@pytest.fixture
def randomMetrics(smallTile, tmp_path):
    syn.writeInputs(tmp_path / 'in', 'h12v02', 2019, _randomInputs)
    return lambda name: syn.metricsRoot(tmp_path / 'in', tmp_path / name)


def test_product_predictors_match_the_c_loops(randomMetrics):
    mt = randomMetrics('metrics')
    names = sorted({n for f in ('predictors_tree.txt', 'predictors_bare.txt')
                    for n in (ROOT / 'docs' / f).read_text().split() if n.startswith('C')})
    assert names
    unsorted = _unsortedFromFile(mt)
    want = {}
    for r in range(syn.SIZE):
        for c in range(syn.SIZE):
            want[(r, c)] = cCompute(unsorted[:, :, r, c].tolist())
    for n in names:
        family, band = n.split('-')
        k = CAttributes.C_BANDS.index(band)
        got = np.asarray(mt.getMetricFromRf(n))
        exp = np.array([[want[(r, c)][family][k] for c in range(syn.SIZE)] for r in range(syn.SIZE)])
        assert np.array_equal(got, exp), n


def test_row_blocks_give_the_same_values(randomMetrics, monkeypatch):
    """computeWanted works in blocks of rows; blocks of 4 rows (4 + 2 on a 6-row tile) equal one block."""
    whole = CM.computeWanted(randomMetrics('a'), CAttributes.wantedFamilies())
    monkeypatch.setattr(CM, 'ROWS_PER_CHUNK', 4)
    blocks = CM.computeWanted(randomMetrics('b'), CAttributes.wantedFamilies())
    assert whole.keys() == blocks.keys()
    for k in whole:
        assert np.array_equal(whole[k], blocks[k]), k


def test_ndvi_period_values(randomMetrics):
    """UnsortedMonthlyBands-NDVI = 1000 (b2 - b1) / (b2 + b1) of the period composites (h12v02 is not a zenith-only NDVI tile)."""
    mt = randomMetrics('metrics')
    b1, x1 = mt.getBandCube('Band_1')
    b2, _ = mt.getBandCube('Band_2')
    for day, i in x1.items():
        got = np.asarray(mt.getMetricFromRf('UnsortedMonthlyBands-NDVI-Day_' + day), float)
        exp = 1000 * (b2[i] - b1[i]) / (b2[i] + b1[i])
        assert np.allclose(got, exp, atol=1), day
