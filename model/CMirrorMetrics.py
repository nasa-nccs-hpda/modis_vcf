
import numpy as np

from modis_vcf.model import CAttributes


# ----------------------------------------------------------------------------
# CMirrorMetrics
#
# Python versions of the metrics that C's make_metrics.c derives from the 72
# unsorted period values (9 bands x 8 periods), computed with C's integer
# arithmetic so the values can be identical to C's.  The statistics are:
#
#   Mean13  (p0 + p1) / 2
#   Mean16  (p0 + p1 + p2 + p3) / 4
#   Mean19  (p0 + p1 + p2 + p3 + p4 + p5) / 6
#   Mean46  (p2 + p3) / 2
#   Mean79  (p4 + p5) / 2
#   Amp15   (p0 + p1 - p2 - p3) / 2
#
# computed over the periods in their unsorted order (family CUns<stat>), and
# over the periods ordered by NDVI (CGreen<stat>, CGreenMax = the first) and
# by Band_31 (CTherm<stat>, CThermMax = the first), both high to low.  Also
# each band sorted on its own, high to low (CSortedRank0 .. CSortedRank7).
#
# C does the sums in int, divides with truncation toward zero, and stores the
# result in a short.
# ----------------------------------------------------------------------------

NDVI_BAND = CAttributes.C_BANDS.index('NDVI')
THERMAL_BAND = CAttributes.C_BANDS.index('Band_31')
ROWS_PER_CHUNK = 400


# ----------------------------------------------------------------------------
# truncDiv
#
# C integer division: truncates toward zero, unlike numpy's floor division.
# ----------------------------------------------------------------------------
def truncDiv(x: np.ndarray, divisor: int) -> np.ndarray:

    return np.where(x >= 0, x // divisor, -((-x) // divisor))


# ----------------------------------------------------------------------------
# statistics
#
# periods is (9, 8, ...) int32; returns {stat: (9, ...)}.
# ----------------------------------------------------------------------------
def statistics(periods: np.ndarray) -> dict:

    p = periods

    return {
        'Mean13': truncDiv(p[:, 0] + p[:, 1], 2),
        'Mean16': truncDiv(p[:, 0] + p[:, 1] + p[:, 2] + p[:, 3], 4),
        'Mean19': truncDiv(p[:, 0] + p[:, 1] + p[:, 2] + p[:, 3] +
                           p[:, 4] + p[:, 5], 6),
        'Mean46': truncDiv(p[:, 2] + p[:, 3], 2),
        'Mean79': truncDiv(p[:, 4] + p[:, 5], 2),
        'Amp15': truncDiv(p[:, 0] + p[:, 1] - p[:, 2] - p[:, 3], 2)}


# ----------------------------------------------------------------------------
# orderByBand
#
# C's exchange sort: whole periods (all nine bands) are swapped so that the
# key band ends up high to low.  Ties are never swapped.
# ----------------------------------------------------------------------------
def orderByBand(periods: np.ndarray, keyBand: int) -> np.ndarray:

    s = periods.copy()

    for n in range(7):
        for nn in range(n + 1, 8):

            swap = s[keyBand, n] < s[keyBand, nn]
            first = s[:, n].copy()
            second = s[:, nn].copy()
            s[:, n] = np.where(swap, second, first)
            s[:, nn] = np.where(swap, first, second)

    return s


# ----------------------------------------------------------------------------
# compute
#
# unsorted is (9, 8, ...) in C band order (Band_1 .. Band_7, NDVI, Band_31)
# and period order.  Returns {family: (9, ...) int16} for every family.
# ----------------------------------------------------------------------------
def compute(unsorted: np.ndarray) -> dict:

    periods = unsorted.astype(np.int32)
    out = {}

    for stat, value in statistics(periods).items():
        out['CUns' + stat] = value

    ranked = -np.sort(-periods, axis=1)

    for rank in range(8):
        out['CSortedRank' + str(rank)] = ranked[:, rank]

    for prefix, key in (('CGreen', NDVI_BAND), ('CTherm', THERMAL_BAND)):

        ordered = orderByBand(periods, key)
        out[prefix + 'Max'] = ordered[:, 0]

        for stat, value in statistics(ordered).items():
            out[prefix + stat] = value

    return {family: value.astype(np.int16) for family, value in out.items()}


# ----------------------------------------------------------------------------
# computeWanted
#
# self is a Metrics.  Reads the unsorted monthly bands and returns
# {(family, band): (rows, cols) int16} for the families and bands in wanted,
# working in blocks of rows so the whole cube is never widened to int32.
# ----------------------------------------------------------------------------
def computeWanted(self, wanted: dict) -> dict:

    # ---
    # Always work from the unsorted values as they are on disk (int16), which is what prediction reads and what was
    # validated against C.  A metric that was just computed holds floats (NDVI is fractional, no-data is NaN), which
    # would give different derived values than the written file.
    # ---
    name = 'metric' + CAttributes.UNSORTED
    self.getMetric(name)                                # computes and writes the file if it is not there yet
    metrics = self.readMetric(self._getOutName(name))
    values = {}
    days = set()

    for m in metrics:

        _, band, day = m.name.split('-')
        values[(band, day)] = m.value
        days.add(day)

    days = sorted(days)

    if len(days) != 8:
        raise RuntimeError('Expected 8 unsorted periods, found ' + str(days))

    rows, cols = values[(CAttributes.C_BANDS[0], days[0])].shape
    result = {(f, b): np.zeros((rows, cols), dtype=np.int16)
              for f, bands in wanted.items() for b in bands}

    for start in range(0, rows, ROWS_PER_CHUNK):

        stop = min(start + ROWS_PER_CHUNK, rows)

        block = np.stack(
            [np.stack([values[(band, day)][start:stop] for day in days])
             for band in CAttributes.C_BANDS])

        derived = compute(block)

        for family, bands in wanted.items():
            for band in bands:
                k = CAttributes.C_BANDS.index(band)
                result[(family, band)][start:stop] = derived[family][k]

    return result


# ----------------------------------------------------------------------------
# install
#
# Adds one metric method per mirror family to the Metrics class, so
# makeMetrics / getMetricFromRf treat them like any other metric.  Each
# family's file holds only the bands the C models use.
# ----------------------------------------------------------------------------
def install(metricsClass, wanted: dict = None) -> None:

    wanted = wanted or CAttributes.wantedFamilies()

    def cache(self) -> dict:

        if getattr(self, '_cMirrorCache', None) is None:

            self._logger.info('Computing C-mirror metrics.')
            self._cMirrorCache = computeWanted(self, wanted)

        return self._cMirrorCache

    def makeMetric(family: str):

        def metric(self) -> list:

            computed = cache(self)
            metrics = []

            for band in wanted[family]:

                name = family + '-' + band
                metrics.append(metricsClass.Metric(name,
                                                   name.replace('-', ' '),
                                                   computed[(family, band)]))

            return metrics

        metric.__name__ = 'metric' + family
        return metric

    for family in wanted:
        setattr(metricsClass, 'metric' + family, makeMetric(family))
