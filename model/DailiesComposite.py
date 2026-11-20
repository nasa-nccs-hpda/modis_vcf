"""48-day VCF composites from C7 daily surface reflectance (B2 rules), for one sensor, tile and composite period.

This is the v2.0.0 replacement for the MOD44C composites: the metrics step reads the files written here exactly as it
reads composites built from MOD44C (<root>/2-Composites/MOD44-<tile>-<yyyyddd>-<band>.bin, int16 4800 x 4800, with the
-qa.npy and -cloudfb.npy flags).  The rules are the Aqua-Terra Experiment's B2 (lead 2026-11-17; B2_Results.md), ported
unchanged from the experiment scripts b0_composite.py, b1_composite.py and b2_composite.py: with the default options the
output is byte for byte that of `b2_composite.py <tile> <root> <out> --polarfill [--sensor aqua]`.

Inputs per day (Sensor: GA 500 m / 1 km, GQ 250 m, TBGA thermal), first layer only:
  reflectance bands 1-2 from GQ layer 1 (250 m), bands 3-7 from GA layer 1 (500 m); state, angles and Band 31 from the 1 km
  observation that GA's 500 m layer 1 came from (iobs_res_1; option (b), lead 2026-11-13); GQ and GA must be the same
  overpass (granule + orbit).  Band 31 is clamped to 32767 before the int16 cast (no wrap).
Good day: present; validity bands (Sensor.validTest) in -100..16000 and Band 31 > 0; same overpass; cloud state 00, no
  internal cloud, no shadow, no adjacency; aerosol low / average (climatology only on snow days); 0 < solar zenith < 72.
  Groups, first non-empty wins: snow-free VZ < 45, snow-free, snow VZ < 45, snow; composite = mean of its days.
Fallback (no good day), first class with any day wins (fbclass diag 1-5):
  F1 soft: no cloud, shadow or high aerosol (rejected by adjacency / climatology) -> mean of the 3 best days
  F2 one-flag: no cloud; shadow or high aerosol -> mean of the 3 best days
  F3 cloudy: cloud or internal cloud, no shadow, NDVI >= 0 and NIR >= 0.15 (v1.9.0 guard) -> mean of the 3 lowest-Red days
  F4: mean of every present day with the low-sun clamp (16000 where solz >= 72 or <= 0)
  5: no present day -> 16000 on the globe (polar-night fill, C1), 10000 off the globe
  "Best" = snow-free first, then lowest view zenith; means use the days with the best day's snow status, truncated to int16.
QA flag (-qa.npy) = no good day; Cloud flag (-cloudfb.npy) = no good day and no day with cloud state 00 under the sun limit.

Options (CompositeOptions), all off or at the B2 value by default; each reproduces its experiment switch:
  polarFill  (B2: on)   C1 polar-night fill 16000; off = B1m3 (10000)
  otherOrbit (off)      C2: a good observation from another orbit before F1 (Terra only; parked 2026-11-16)
  f2Rank     (off)      C3: F2 days without the shadow flag rank first (dropped 2026-11-16)
  f3         ('mean3')  'single' = F3 is the single lowest-Red day (B1 v1)
  f3Guard    (on)       off = no NDVI / NIR guard on F3 days (B1 v1, v2)
  rules      ('b1')     'b0' = B0's rules: sun limit 65, clamp above 70, no F1-F3 classes (all-days mean)
"""

import datetime
import glob
import json
import os
import time
from dataclasses import dataclass, asdict

import h5py
import numpy as np

from modis_vcf.model.Sensor import Sensor, TERRA

N = 4800
M = N // 2
BANDS = ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7', 'Band_31']
NB = len(BANDS)
VZ_PREF, CLAMP, SUBST, FILL = 45.0, 16000, 10000, -28672
BIG = np.uint32(0xFFFFFFFF)
NX = 3                                     # extra rows kept with each fallback day: YYDDD, solz x100, vz x100
R_EARTH = 6371007.181
T_SIZE = 1111950.5197665
X0, Y0 = -20015109.354, 10007554.677
PERIOD_DAYS = (65, 113, 161, 209, 257, 305, 353)   # then 033 of the next year (Metrics._daysSought)
ESSENTIAL_DIAG = ('fbclass', 'b6_n')        # written even with diag off (AquaFill reads them)
DIAG_COUNTS = ('snow15_only', 'snow12_only', 'snow_both', 'clim', 'mismatch', 'b31_over', 'qa_not_layer1')


@dataclass(frozen=True)
class CompositeOptions:
    """The compositor's switches (module docstring).  The defaults are B2."""
    polarFill: bool = True
    otherOrbit: bool = False
    f2Rank: bool = False
    f3: str = 'mean3'
    f3Guard: bool = True
    rules: str = 'b1'

    def __post_init__(self):
        if self.f3 not in ('single', 'mean3'):
            raise ValueError('f3 must be single or mean3')
        if self.rules not in ('b0', 'b1'):
            raise ValueError('rules must be b0 or b1')


# ----------------------------------------------------------------------------
# periods and days
# ----------------------------------------------------------------------------
def periods(year: int) -> list:
    """The 8 composite periods of a processing year as 'yyyyddd' (065 ... 353, then 033 of the next year)."""
    return ['%d%03d' % (year, d) for d in PERIOD_DAYS] + ['%d033' % (year + 1)]


def processingYear(period: str) -> int:
    """The processing year a period belongs to (033 is in the next calendar year)."""
    y, d = int(period[:4]), int(period[4:])
    return y - 1 if d < PERIOD_DAYS[0] else y


def daysOf(period: str) -> list:
    """Every day from the period start to the day before the next period; the last period (033) runs to DOY 064 of its
    calendar year, the end of the processing year.  These are the days of v1.9.0's 16-day MOD44C inputs of the period:
    MOD44C restarts on 1 January, so period 353 is 353-032 (45 or 46 days) and 033 is 033-064 (32 days)."""
    ps = periods(processingYear(period))
    i = ps.index(period)
    d0 = datetime.datetime.strptime(period, '%Y%j').date()
    end = datetime.datetime.strptime(ps[i + 1], '%Y%j').date() if i + 1 < len(ps) else \
        datetime.datetime.strptime('%s064' % period[:4], '%Y%j').date() + datetime.timedelta(1)
    return [d0 + datetime.timedelta(k) for k in range((end - d0).days)]


def findDaily(root: str, product: str, tile: str, day: datetime.date, collection: str = '007') -> str:
    """The newest daily file of a product, tile and day under <root>/<product>/<year>/, or None."""
    f = glob.glob('%s/%s/%d/%s.A%d%03d.%s.%s.*.h5' % (root, product, day.year, product, day.year,
                                                     day.timetuple().tm_yday, tile, collection))
    return sorted(f)[-1] if f else None


def hasDailies(root: str, sensor: Sensor, tile: str, period: str) -> bool:
    """True when the period has at least one day with all three of the sensor's daily files for the tile."""
    return any(all(findDaily(root, p, tile, d, sensor.collection) for p in sensor.products) for d in daysOf(period))


# ----------------------------------------------------------------------------
# reading one day
# ----------------------------------------------------------------------------
def up(a, k):
    return np.repeat(np.repeat(a, k, 0), k, 1)


class Pick1km:
    """The 1 km fields (state, angles, Band 31) of the 1 km observation each 500 m layer-1 observation came from.
    iobs_res_1 (500 m) = index of that observation in the 1 km pixel's list: 0 = the 1 km layer 1 (`<name>_1`), k >= 1 =
    the k-th additional layer in the compact array `<name>_c` (row by row, pixel by pixel; checked 2026-11-13).  GA and
    TBGA have the same 1 km observation lists, so the same index serves both."""
    def __init__(self, nobs1km, iobs500):
        nadd = np.maximum(nobs1km.astype(np.int64) - 1, 0)
        off = np.concatenate([[0], np.cumsum(nadd.ravel())])[:-1].reshape(nadd.shape)
        self.k = iobs500.astype(np.int64)
        self.fill = self.k == 255
        self.layer1 = self.k == 0
        self.idx = np.where(self.k >= 1, up(off, 2) + self.k - 1, 0)
        self.bad = ~self.fill & (self.k >= 1) & (self.k > up(nadd, 2))       # pointer past the pixel's list: missing

    def take(self, layer1, compact, fill):
        v = np.where(self.layer1, up(layer1, 2), compact[np.minimum(self.idx, len(compact) - 1)] if len(compact) else fill)
        return np.where(self.fill | self.bad, fill, v).astype(layer1.dtype)


def readDay(root: str, tile: str, day: datetime.date, sensor: Sensor) -> tuple:
    """(fields on the 250 m grid, {product: file}); fields is None when a file is missing."""
    fs = {p: findDaily(root, p, tile, day, sensor.collection) for p in sensor.products}

    if None in fs.values():
        return None, fs

    G5, G1, GQ, TB = sensor.group500m, sensor.group1km, sensor.group250m, sensor.groupThermal
    ga, gq, tb = (h5py.File(fs[p], 'r') for p in sensor.products)
    d = {}
    d['Band_1'] = gq[GQ + 'sur_refl_b01_1'][:]
    d['Band_2'] = gq[GQ + 'sur_refl_b02_1'][:]
    assert d['Band_1'].shape == (N, N), d['Band_1'].shape

    for i in range(3, 8):
        d['Band_%d' % i] = up(ga[G5 + 'sur_refl_b%02d_1' % i][:], 2)

    pk = Pick1km(ga[G1 + 'num_observations_1km'][:], ga[G5 + 'iobs_res_1'][:])
    t = pk.take(tb[TB + 'BAND31_1'][:], tb['BAND31_c'][:], 0)
    d['b31_over'] = up(t > 32767, 2)
    d['Band_31'] = up(np.minimum(t, 32767).astype(np.int16), 2)
    d['state'] = up(pk.take(ga[G1 + 'state_1km_1'][:], ga['state_1km_c'][:], 65535), 2)
    d['solz'] = up(pk.take(ga[G1 + 'SolarZenith_1'][:], ga['SolarZenith_c'][:], -32767), 2)      # x0.01 deg
    d['vz'] = up(pk.take(ga[G1 + 'SensorZenith_1'][:], ga['SensorZenith_c'][:], -32767), 2)
    d['layer1_1km'] = up(pk.layer1, 2)
    d['qc500'] = up(ga[G5 + 'QC_500m_1'][:], 2)
    # Same overpass: GQ's granule and orbit equal those of GA's 1 km observation (the index may differ; the granule not).
    gran = up(pk.take(ga[G1 + 'granule_pnt_1'][:], ga['granule_pnt_c'][:], 255), 2)
    orb = up(pk.take(ga[G1 + 'orbit_pnt_1'][:], ga['orbit_pnt_c'][:], -1), 2)
    d['same'] = up(~(pk.fill | pk.bad), 2) & (gq[GQ + 'granule_pnt_1'][:] == gran) & \
        (gq[GQ + 'orbit_pnt_1'][:] == orb) & (gran != 255)

    for h in (ga, gq, tb):
        h.close()

    return d, fs


# ----------------------------------------------------------------------------
# per-day tests
# ----------------------------------------------------------------------------
def masks(d: dict, sensor: Sensor, rules: str = 'b1', f3guard: bool = True) -> dict:
    """The good-day test, groups, low-sun clamp and fallback classes of one day (module docstring)."""
    zen, lows = (65.0, 70.0) if rules == 'b0' else (72.0, 72.0)
    s = d['state'].astype(np.int32)
    solz = d['solz'] / 100.0
    vz = d['vz'] / 100.0
    present = (d['state'] != 65535) & (d['solz'] != -32767)
    valid = np.logical_and.reduce([(d[b] >= -100) & (d[b] <= 16000) for b in sensor.validTest]) & (d['Band_31'] != 0)
    same = d['same']
    snow = ((s >> 15) & 1 == 1) | ((s >> 12) & 1 == 1)
    aer = (s >> 6) & 3
    clear = ((s & 3) == 0) & ((s >> 10) & 1 == 0) & ((s >> 2) & 1 == 0) & ((s >> 13) & 1 == 0)
    aerOk = (aer == 1) | (aer == 2) | ((aer == 0) & snow)
    sun = (solz > 0) & (solz < zen)
    good = present & valid & same & clear & aerOk & sun
    m = dict(present=present, good=good, solz=solz, vz=vz,
             group=np.where(~snow, np.where(vz < VZ_PREF, 0, 1), np.where(vz < VZ_PREF, 2, 3)),
             cloudclear=present & ((s & 3) == 0) & (solz < zen),
             snow15=((s >> 15) & 1 == 1), snow12=((s >> 12) & 1 == 1), clim=present & (aer == 0),
             mismatch=present & ~same, snow=snow)
    m['lowsun'] = present & ((solz > lows) | (solz <= 0)) if rules == 'b0' else present & ((solz >= lows) | (solz <= 0))
    base = present & valid & same & (solz > 0) & (solz < zen)
    cloud = ((s & 3) != 0) | ((s >> 10) & 1 == 1)
    shadow = (s >> 2) & 1 == 1
    high = aer == 3
    m['f1'] = base & ~good & ~cloud & ~shadow & ~high
    m['f2'] = base & ~good & ~cloud & (shadow | high)
    m['f3'] = base & ~good & cloud & ~shadow

    if f3guard:
        r1 = d['Band_1'].astype(np.int32)
        r2 = d['Band_2'].astype(np.int32)
        m['f3'] &= (r2 >= r1) & (r2 >= 1500)               # NDVI >= 0 and NIR >= 0.15, v1.9.0's pick guard

    return m


class Best3:
    """Per pixel, the 3 days with the smallest key (uint32), with their band values."""
    def __init__(self):
        self.k = np.full((3, N, N), BIG, np.uint32)
        self.v = np.zeros((3, NB + NX, N, N), np.int16)

    def add(self, sel, key, vals):
        key = np.where(sel, key, BIG).astype(np.uint32)
        k0, k1, k2 = self.k
        p0 = key < k0
        p1 = ~p0 & (key < k1)
        p2 = ~p0 & ~p1 & (key < k2)
        v0, v1, v2 = self.v[0].copy(), self.v[1].copy(), self.v[2].copy()
        self.k[2] = np.where(p0 | p1, k1, np.where(p2, key, k2))
        self.v[2] = np.where((p0 | p1)[None], v1, np.where(p2[None], vals, v2))
        self.k[1] = np.where(p0, k0, np.where(p1, key, k1))
        self.v[1] = np.where(p0[None], v0, np.where(p1[None], vals, v1))
        self.k[0] = np.where(p0, key, k0)
        self.v[0] = np.where(p0[None], vals, v0)

    def meanSameSnow(self, shift):
        """Mean of the kept days whose snow status (key >> shift) equals the best day's; (values float64, any)."""
        ok = (self.k != BIG) & ((self.k >> shift) == (self.k[0] >> shift)[None])
        n = ok.sum(0)
        s = (self.v[:, :NB].astype(np.int64) * ok[:, None]).sum(0)
        return s / np.where(n > 0, n, 1), n > 0

    def used(self, shift, k):
        """Mask (3, N, N) of the days that enter the fill: the best k kept days with the best day's snow status."""
        ok = (self.k != BIG) & ((self.k >> shift) == (self.k[0] >> shift)[None])
        ok[k:] = False
        return ok


# ----------------------------------------------------------------------------
# C1 and C2 helpers
# ----------------------------------------------------------------------------
def onGlobe(tile: str) -> np.ndarray:
    """True where the 250 m pixel centre lies on the sinusoidal Earth."""
    h, v = int(tile[1:3]), int(tile[4:6])
    px = T_SIZE / N
    x = X0 + h * T_SIZE + (np.arange(N) + 0.5) * px
    y = Y0 - v * T_SIZE - (np.arange(N) + 0.5) * px
    return np.abs(x)[None, :] <= np.pi * R_EARTH * np.cos(y / R_EARTH)[:, None]


def _offsets(nobs):
    """Start of each pixel's additional observations in a compact `*_c` array (row by row, pixel by pixel)."""
    nadd = np.maximum(nobs.astype(np.int64) - 1, 0)
    return np.concatenate([[0], np.cumsum(nadd.ravel())])[:-1].reshape(nobs.shape), nadd


def _gather(a1, ac, nobs, off, j, fill):
    """Per pixel, its j-th observation (0 = layer 1, >= 1 = compact `*_c`, < 0 = none); fill where j < 0 or j >= nobs."""
    j = np.asarray(j, np.int64)

    if len(ac):
        v = np.where(j == 0, a1, ac[np.clip(off + j - 1, 0, len(ac) - 1)])
    else:
        v = np.where(j == 0, a1, fill)

    return np.where((j >= 0) & (j < nobs), v, fill).astype(a1.dtype)


def _goodQa(s, solz):
    """The good test on QA and sun only (state uint16, solz in degrees); (good, snow)."""
    s = s.astype(np.int32)
    snow = ((s >> 15) & 1 == 1) | ((s >> 12) & 1 == 1)
    aer = (s >> 6) & 3
    clear = ((s & 3) == 0) & ((s >> 10) & 1 == 0) & ((s >> 2) & 1 == 0) & ((s >> 13) & 1 == 0)
    return (s != 65535) & clear & ((aer == 1) | (aer == 2) | ((aer == 0) & snow)) & (solz > 0) & (solz < 72.0), snow


def otherOrbitDay(fs: dict, sensor: Sensor) -> dict:
    """C2 for one day: 250 m arrays (bands, state, solz, vz, ok, snow) of the chosen other-orbit observation, or None.
    Per 500 m pixel the compact observation that passes QA + sun, comes from another orbit than layer 1 and has the lowest
    view zenith; bands 1-2 from the GQ observation of the same granule and orbit with the highest obscov."""
    G5, G1, GQ, TB = sensor.group500m, sensor.group1km, sensor.group250m, sensor.groupThermal

    with h5py.File(fs[sensor.ga], 'r') as ga, h5py.File(fs[sensor.gq], 'r') as gq, h5py.File(fs[sensor.tbga], 'r') as tb:
        n5 = ga[G5 + 'num_observations_500m'][:].astype(np.int64)

        if n5.max() < 2:
            return None

        off5, _ = _offsets(n5)
        n1 = ga[G1 + 'num_observations_1km'][:].astype(np.int64)
        off1, _ = _offsets(n1)
        n1u, off1u = up(n1, 2), up(off1, 2)
        f1 = {k: (up(ga[G1 + k + '_1'][:], 2), ga[k + '_c'][:])
              for k in ('state_1km', 'SolarZenith', 'SensorZenith', 'orbit_pnt', 'granule_pnt')}
        f1['BAND31'] = (up(tb[TB + 'BAND31_1'][:], 2), tb['BAND31_c'][:])
        fill1 = dict(state_1km=65535, SolarZenith=-32767, SensorZenith=-32767, orbit_pnt=-1, granule_pnt=255, BAND31=0)
        at1 = lambda name, k: _gather(f1[name][0], f1[name][1], n1u, off1u, k, fill1[name])
        k5_1 = ga[G5 + 'iobs_res_1'][:].astype(np.int64)
        k5_c = ga['iobs_res_c'][:]
        orbL1 = at1('orbit_pnt', np.where(k5_1 == 255, -1, k5_1)).astype(np.int64)
        bestVz = np.full((M, M), 32767, np.int32)
        bestJ = np.full((M, M), -1, np.int64)

        for j in range(1, int(n5.max())):
            jj = np.where(j < n5, j, -1)
            k = _gather(k5_1, k5_c, n5, off5, jj, 255).astype(np.int64)
            k = np.where(k == 255, -1, k)

            if not (k >= 0).any():
                continue

            g, _ = _goodQa(at1('state_1km', k), at1('SolarZenith', k) / 100.0)
            vz = at1('SensorZenith', k).astype(np.int32)
            o = at1('orbit_pnt', k).astype(np.int64)
            c = (k >= 0) & g & (o >= 0) & (o != orbL1) & (vz >= 0) & (vz < bestVz)
            bestVz = np.where(c, vz, bestVz)
            bestJ = np.where(c, j, bestJ)

        chosen = bestJ >= 1

        if not chosen.any():
            return None

        k = _gather(k5_1, k5_c, n5, off5, bestJ, 255).astype(np.int64)
        k = np.where(chosen & (k != 255), k, -1)
        s, solz, vz = at1('state_1km', k), at1('SolarZenith', k), at1('SensorZenith', k)
        orb, gran, b31 = at1('orbit_pnt', k).astype(np.int64), at1('granule_pnt', k).astype(np.int64), at1('BAND31', k)
        d = {}

        for i in (3, 4, 5, 6, 7):
            d['Band_%d' % i] = up(_gather(ga[G5 + 'sur_refl_b%02d_1' % i][:], ga['sur_refl_b%02d_c' % i][:], n5, off5,
                                          bestJ, FILL), 2)

        idx = np.flatnonzero(up(chosen, 2).ravel())
        nq = gq[GQ + 'num_observations'][:].astype(np.int64)
        offq, _ = _offsets(nq)
        nqS, offqS = nq.ravel()[idx], offq.ravel()[idx]
        ou, gu = up(orb, 2).ravel()[idx], up(gran, 2).ravel()[idx]
        q = {k_: (gq[GQ + k_ + '_1'][:].ravel()[idx], gq[k_ + '_c'][:])
             for k_ in ('orbit_pnt', 'granule_pnt', 'obscov', 'sur_refl_b01', 'sur_refl_b02')}
        qfill = dict(orbit_pnt=-1, granule_pnt=255, obscov=-1, sur_refl_b01=FILL, sur_refl_b02=FILL)
        bestc = np.full(idx.size, -1, np.int16)
        bq1 = np.full(idx.size, FILL, np.int16)
        bq2 = bq1.copy()

        for j in range(int(nqS.max()) if idx.size else 0):
            jj = np.full(idx.size, j, np.int64)
            gq_ = lambda n_: _gather(q[n_][0], q[n_][1], nqS, offqS, jj, qfill[n_])
            o, g_, cv = gq_('orbit_pnt').astype(np.int64), gq_('granule_pnt').astype(np.int64), gq_('obscov').astype(np.int16)
            mm = (o == ou) & (g_ == gu) & (gu != 255) & (cv > bestc)

            if mm.any():
                bestc = np.where(mm, cv, bestc)
                bq1 = np.where(mm, gq_('sur_refl_b01'), bq1)
                bq2 = np.where(mm, gq_('sur_refl_b02'), bq2)

    for b, v in (('Band_1', bq1), ('Band_2', bq2)):
        a = np.full(N * N, FILL, np.int16)
        a[idx] = v
        d[b] = a.reshape(N, N)

    matched = np.zeros(N * N, bool)
    matched[idx] = bestc >= 0
    matched = matched.reshape(N, N)
    d['Band_31'] = up(np.minimum(b31, 32767).astype(np.int16), 2)
    d['state'] = up(s, 2)
    d['solz'] = up(solz, 2)
    d['vz'] = up(vz, 2)
    valid = np.logical_and.reduce([(d[b] >= -100) & (d[b] <= 16000) for b in sensor.validTest]) & (d['Band_31'] != 0)
    g, snow = _goodQa(d['state'], d['solz'] / 100.0)
    d['ok'] = up(chosen, 2) & matched & valid & g
    d['snow'] = snow
    return d


# ----------------------------------------------------------------------------
# composite
# ----------------------------------------------------------------------------
def compositeName(out: str, tile: str, period: str, band: str, suffix: str = '.bin') -> str:
    """<out>/2-Composites/MOD44-<tile>-<period>-<band><suffix> (CompositeDayFile's names, whatever the sensor)."""
    return '%s/2-Composites/MOD44-%s-%s-%s%s' % (out, tile, period, band, suffix)


def diagName(out: str, tile: str, period: str, name: str) -> str:
    """<out>/diag/<tile>-<period>-<name>.npy (summary.json when name ends in .json)."""
    return '%s/diag/%s-%s-%s' % (out, tile, period, name if name.endswith('.json') else name + '.npy')


def isDone(out: str, tile: str, period: str) -> bool:
    """A period is complete when its summary.json exists (written last)."""
    return os.path.exists(diagName(out, tile, period, 'summary.json'))


def composite(tile: str, period: str, root: str, out: str, sensor: Sensor = TERRA,
              options: CompositeOptions = CompositeOptions(), diag: bool = False) -> dict:
    """Build one period's composites (8 bands, flags) into out; returns the period summary (diag/summary.json).
    diag: also write the per-pixel diagnostics (about 0.5 GB per period); off by default (lead 2026-11-19).  Always written:
    fbclass and, for Aqua, b6_n (the Aqua fill needs them) and summary.json.  The composites are the same either way."""
    o = options
    rules, f3, f3guard, polarfill, otherorbit, f2rank = o.rules, o.f3, o.f3Guard, o.polarFill, o.otherOrbit, o.f2Rank
    b1 = rules == 'b1'
    aqua = sensor.band6ValidDaysOnly
    I6 = BANDS.index('Band_6')

    if otherorbit and (aqua or not b1):
        raise ValueError('otherOrbit (C2) is defined for Terra with the b1 rules only')

    if aqua:
        b6sum = np.zeros((4, N, N), np.int32)
        b6cnt = np.zeros((4, N, N), np.int16)

    sums = np.zeros((4, NB, N, N), np.int32)
    cnt = np.zeros((4, N, N), np.int16)
    fsum = np.zeros((NB, N, N), np.int32)
    fcnt = np.zeros((NB, N, N), np.int16)
    anyclear = np.zeros((N, N), bool)
    npres = np.zeros((N, N), np.int16)
    zs = np.zeros((4, N, N), np.float32)
    vs = np.zeros((4, N, N), np.float32)

    if otherorbit:
        osums = np.zeros((4, NB, N, N), np.int32)
        ocnt = np.zeros((4, N, N), np.int16)
        ozs = np.zeros((4, N, N), np.float32)
        ovs = np.zeros((4, N, N), np.float32)

    dg = {k: np.zeros((N, N), np.int16) for k in DIAG_COUNTS}
    F1, F2, F3 = (Best3(), Best3(), Best3()) if b1 else (None, None, None)
    log, inputs = [], []
    T = dict(read=0.0, accumulate=0.0, other=0.0, write=0.0)

    for day in daysOf(period):
        t = time.time()
        d, fs = readDay(root, tile, day, sensor)
        T['read'] += time.time() - t
        t = time.time()

        if d is None:
            log.append('%s missing %s' % (day, [p for p, f in fs.items() if f is None]))
            continue

        inputs += [os.path.basename(f) for f in fs.values()]
        m = masks(d, sensor, rules, f3guard)
        p = m['present']
        npres += p
        anyclear |= m['cloudclear']
        dg['snow15_only'] += p & m['snow15'] & ~m['snow12']
        dg['snow12_only'] += p & m['snow12'] & ~m['snow15']
        dg['snow_both'] += p & m['snow15'] & m['snow12']
        dg['clim'] += m['clim']
        dg['mismatch'] += m['mismatch']
        dg['b31_over'] += p & d['b31_over']
        dg['qa_not_layer1'] += p & ~d['layer1_1km']

        for g in range(4):
            sel = m['good'] & (m['group'] == g)
            cnt[g] += sel
            zs[g] += np.where(sel, m['solz'], 0)
            vs[g] += np.where(sel, m['vz'], 0)

            for i, b in enumerate(BANDS):
                sums[g, i] += np.where(sel, d[b], 0)

            if aqua:
                s6 = sel & (d['Band_6'] >= -100) & (d['Band_6'] <= 16000)
                b6sum[g] += np.where(s6, d['Band_6'], 0)
                b6cnt[g] += s6

        for i, b in enumerate(BANDS):
            v = np.where(m['lowsun'], CLAMP, d[b]).astype(np.int32)
            ok = p & (d[b] != FILL) & (d[b] != 0 if b == 'Band_31' else True)
            fsum[i] += np.where(ok, v, 0)
            fcnt[i] += ok

        if b1:
            yd = np.int16(int(day.strftime('%y%j')))
            vals = np.concatenate([np.stack([d[b] for b in BANDS]).astype(np.int16), np.full((1, N, N), yd, np.int16),
                                   np.stack([d['solz'], d['vz']]).astype(np.int16)])
            sn = m['snow'].astype(np.uint32)
            vzk = np.clip(np.rint(m['vz']), 0, 255).astype(np.uint32)
            F1.add(m['f1'], (sn << 8) | vzk, vals)

            if f2rank:
                shd = (((d['state'].astype(np.int32) >> 2) & 1) == 1).astype(np.uint32)
                F2.add(m['f2'], (sn << 9) | (shd << 8) | vzk, vals)
            else:
                F2.add(m['f2'], (sn << 8) | vzk, vals)

            F3.add(m['f3'], (sn << 16) | np.clip(d['Band_1'].astype(np.int32) + 100, 0, 65535).astype(np.uint32), vals)
            del vals

        T['accumulate'] += time.time() - t
        del d, m

        if otherorbit:
            t = time.time()
            oo = otherOrbitDay(fs, sensor)

            if oo is not None:
                vz = oo['vz'] / 100.0
                sz = oo['solz'] / 100.0
                grp = np.where(~oo['snow'], np.where(vz < VZ_PREF, 0, 1), np.where(vz < VZ_PREF, 2, 3))

                for g in range(4):
                    sel = oo['ok'] & (grp == g)
                    ocnt[g] += sel
                    ozs[g] += np.where(sel, sz, 0)
                    ovs[g] += np.where(sel, vz, 0)

                    for i, b in enumerate(BANDS):
                        osums[g, i] += np.where(sel, oo[b], 0)

            del oo
            T['other'] += time.time() - t

    t = time.time()
    first = np.full((N, N), -1, np.int8)

    for g in (3, 2, 1, 0):
        first = np.where(cnt[g] > 0, g, first)

    fb = first < 0
    cls = np.where(fb, np.where(npres > 0, 4, 5), 0).astype(np.int8)

    if b1:
        m1, a1 = F1.meanSameSnow(8)
        m2, a2 = F2.meanSameSnow(8)
        a3 = F3.k[0] != BIG

        if f3 == 'mean3':
            m3, _ = F3.meanSameSnow(16)

        cls = np.where(fb & a1, 1, np.where(fb & ~a1 & a2, 2, np.where(fb & ~a1 & ~a2 & a3, 3, cls))).astype(np.int8)

    if otherorbit:
        ofirst = np.full((N, N), -1, np.int8)

        for g in (3, 2, 1, 0):
            ofirst = np.where(ocnt[g] > 0, g, ofirst)

        cls = np.where(fb & (ofirst >= 0), 6, cls).astype(np.int8)
        ogi = np.clip(ofirst, 0, 3).astype(np.intp)
        oc = np.take_along_axis(ocnt, ogi[None], 0)[0].astype(np.float64)

    os.makedirs(out + '/2-Composites', exist_ok=True)
    os.makedirs(out + '/diag', exist_ok=True)
    gi = np.clip(first, 0, 3).astype(np.intp)
    c = np.take_along_axis(cnt, gi[None], 0)[0].astype(np.float64)
    globe = onGlobe(tile) if polarfill else None
    fbq = fb & (cls != 6)                                   # class 6 = good observations: qa / cloudfb 0
    fclasses = ((1, F1, 8, 3), (2, F2, 8, 3), (3, F3, 16, 3 if f3 == 'mean3' else 1)) if b1 else ()

    for i, b in enumerate(BANDS):
        good = np.take_along_axis(sums[:, i], gi[None], 0)[0] / np.where(c > 0, c, 1)
        nofill = np.where(globe, CLAMP, SUBST) if polarfill else SUBST
        fall = np.where(fcnt[i] > 0, fsum[i] / np.where(fcnt[i] > 0, fcnt[i], 1), nofill)

        if b1:
            fall = np.where(cls == 1, m1[i], np.where(cls == 2, m2[i],
                            np.where(cls == 3, m3[i] if f3 == 'mean3' else F3.v[0, i], fall)))

        if otherorbit:
            fall = np.where(cls == 6, np.take_along_axis(osums[:, i], ogi[None], 0)[0] / np.where(oc > 0, oc, 1), fall)

        if aqua and i == I6:                                # Band 6 from valid days only (Aqua rule (i))
            c6n = np.take_along_axis(b6cnt, gi[None], 0)[0]
            good = np.where(c6n > 0, np.take_along_axis(b6sum, gi[None], 0)[0] / np.where(c6n > 0, c6n, 1), FILL)
            b6n = np.where(fb, 0, c6n).astype(np.int16)

            for c_, B, sh, k in fclasses:
                v6 = B.v[:, I6].astype(np.int64)
                u6 = B.used(sh, k) & (v6 >= -100) & (v6 <= 16000)
                n6 = u6.sum(0)
                fall = np.where(cls == c_, np.where(n6 > 0, (v6 * u6).sum(0) / np.where(n6 > 0, n6, 1), FILL), fall)
                b6n = np.where(cls == c_, n6, b6n)

            fall = np.where((cls == 4) & (fcnt[i] == 0), FILL, fall)
            b6n = np.where(cls == 4, fcnt[i], b6n).astype(np.int16)

        comp = np.where(fb, fall, good).astype(np.int16)
        comp.tofile(compositeName(out, tile, period, b))
        np.save(compositeName(out, tile, period, b, '-qa.npy'), fbq.astype(np.uint8))
        np.save(compositeName(out, tile, period, b, '-cloudfb.npy'), (fbq & ~anyclear).astype(np.uint8))

    dn = lambda k: diagName(out, tile, period, k)
    sv = lambda k, v: np.save(dn(k), v) if diag or k in ESSENTIAL_DIAG else None
    sv('group', (first + 1).astype(np.int8))
    sv('fbclass', cls)
    sv('gooddays', cnt.sum(0).astype(np.int16))
    sv('present', npres)
    su = np.where(fb, np.nan, np.take_along_axis(zs, gi[None], 0)[0] / np.where(c > 0, c, 1))
    vu = np.where(fb, np.nan, np.take_along_axis(vs, gi[None], 0)[0] / np.where(c > 0, c, 1))

    if otherorbit:
        su = np.where(cls == 6, np.take_along_axis(ozs, ogi[None], 0)[0] / np.where(oc > 0, oc, 1), su)
        vu = np.where(cls == 6, np.take_along_axis(ovs, ogi[None], 0)[0] / np.where(oc > 0, oc, 1), vu)
        sv('other_n', np.where(cls == 6, oc, 0).astype(np.int16))

    if aqua:
        sv('b6_n', b6n)

    sv('solz_used', su.astype(np.float32))
    sv('vz_used', vu.astype(np.float32))

    for k, v in dg.items():
        sv(k, v)

    if b1 and diag:
        _saveFallbackDiag(dn, cls, fclasses, f2rank)

        if f2rank:
            sv('f2_shadow', np.where(cls == 2, (F2.k[0] >> 8) & 1, 0).astype(np.int8))

    T['write'] = time.time() - t
    summ = dict(tile=tile, period=period, sensor=sensor.name, rules='b2', options=asdict(o), days=len(daysOf(period)),
                days_missing=log, fallback_share=float(fb.mean()),
                fbclass_share={k: float((cls == k).mean()) for k in range(7)},
                cloudfb_share=float((fbq & ~anyclear).mean()), group_share=[float((first == g).mean()) for g in range(4)],
                mismatch_pixel_days=int(dg['mismatch'].sum()), present_pixel_days=int(npres.sum()),
                b31_over_pixel_days=int(dg['b31_over'].sum()), qa_not_layer1_pixel_days=int(dg['qa_not_layer1'].sum()),
                inputs=inputs, diag=diag, seconds=T)
    _writeJson(dn('summary.json'), summ)
    return summ


def _saveFallbackDiag(dn, cls, fclasses, f2rank):
    """fb_* diag of the F1-F3 pixels: days in the fill, their YYDDD / solz / vz, snow status and the spread of the days."""
    z = np.zeros((N, N), np.int16)
    nd_ = z.copy()
    dd = np.zeros((3, N, N), np.int16)
    sz = dd.copy()
    vz = dd.copy()
    snf = np.zeros((N, N), np.int8)
    rr, nr, vr = z.copy(), z.copy(), z.copy()

    for c_, B, sh, k in fclasses:
        w = cls == c_
        u = B.used(sh, k)
        kept = B.k != BIG
        nd_ = np.where(w, u.sum(0), nd_)
        snf = np.where(w, (B.k[0] >> (sh + 1 if (c_ == 2 and f2rank) else sh)) & 1, snf)

        for j in range(3):
            dd[j] = np.where(w & kept[j], B.v[j, NB], dd[j])
            sz[j] = np.where(w & kept[j], B.v[j, NB + 1], sz[j])
            vz[j] = np.where(w & kept[j], B.v[j, NB + 2], vz[j])

        r = B.v[:, 0].astype(np.float32)
        n_ = B.v[:, 1].astype(np.float32)
        ndv = (n_ - r) / np.maximum(n_ + r, 1)
        rng = lambda x: np.where(u, x, -np.inf).max(0) - np.where(u, x, np.inf).min(0)
        anyu = u.any(0) & w
        rr = np.where(anyu, rng(r), rr)
        nr = np.where(anyu, rng(n_), nr)
        vr = np.where(anyu, np.clip(rng(ndv) * 1e4, 0, 32767), vr)

    for k_, v_ in (('fb_ndays', nd_.astype(np.int8)), ('fb_days', dd), ('fb_solz', sz), ('fb_vz', vz),
                   ('fb_snow', snf.astype(np.int8)), ('fb_red_range', rr.astype(np.int16)),
                   ('fb_nir_range', nr.astype(np.int16)), ('fb_ndvi_range', vr.astype(np.int16))):
        np.save(dn(k_), v_)


def _writeJson(path: str, obj: dict) -> None:
    """Write under a temporary name, then rename, so a half-written summary never marks a period done."""
    tmp = path + '.part'

    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)

    os.replace(tmp, path)
