import logging
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.CompositeDayFile import CompositeDayFile
from modis_vcf.model.Metrics import Metrics


# -----------------------------------------------------------------------------
# class MetricsZenithOnlyNdviTestCase
#
# Checks the zenith-only NDVI step on a 2 x 2 tile with fake day files, so it
# needs no MODIS input.
#
# python -m unittest modis_vcf.model.tests.test_Metrics_zenithOnlyNdvi
# -----------------------------------------------------------------------------
class MetricsZenithOnlyNdviTestCase(unittest.TestCase):

    # Two periods of two days each.  Solz per day; None means a missing file.
    DAYS = {(2019, 65): [(2019, 65), (2019, 81)],
            (2019, 113): [(2019, 113), (2019, 129)]}

    SOLZ = {(2019, 65): np.array([[70, 64], [0, 80]]),
            (2019, 81): np.array([[66, 90], [0, 30]]),
            (2019, 113): None,
            (2019, 129): np.array([[10, 65], [70, 1]])}

    # -------------------------------------------------------------------------
    # makeMetrics
    # -------------------------------------------------------------------------
    @staticmethod
    def _makeMetrics(tid):

        m = Metrics.__new__(Metrics)
        m._tid = tid
        m._daysSought = list(MetricsZenithOnlyNdviTestCase.DAYS)
        m._productType = SimpleNamespace(ROWS=2, COLS=2, BAND1='Band_1')
        m._compDir = m._dayDir = None
        m._logger = logging.getLogger('test')
        return m

    # -------------------------------------------------------------------------
    # fakeCdf / fakeBdf
    # -------------------------------------------------------------------------
    def _fakeCdf(self, productType, bandName, tid, year, day, compDir,
                 logger=None, dayDir=None):

        days = MetricsZenithOnlyNdviTestCase.DAYS[(year, day)]
        return SimpleNamespace(_getDaysToFind=lambda: days)

    def _fakeBdf(self, productType, bandName, tid, year, day, dayDir, logger):

        solz = MetricsZenithOnlyNdviTestCase.SOLZ[(year, day)]

        def getSolz():
            if solz is None:
                raise RuntimeError('Unable to find file')
            return solz

        return SimpleNamespace(getSolz=getSolz)

    # -------------------------------------------------------------------------
    # testApplyZenithOnlyNdvi
    # -------------------------------------------------------------------------
    def testApplyZenithOnlyNdvi(self):

        ndvi = np.full((2, 2, 2), 500.0)
        m = self._makeMetrics('h16v01')

        with mock.patch.object(CompositeDayFile, 'initFromParams',
                               new=self._fakeCdf), \
             mock.patch.object(BandDayFile, 'initFromParams',
                               new=self._fakeBdf):

            out = m._applyZenithOnlyNdvi(ndvi)

        # Period 0: valid where any day has 0 < solz < 65 (solz 0 is not).
        np.testing.assert_array_equal(out[0], [[0, 500], [0, 500]])

        # Period 1: the missing day counts as no valid zenith; 65 is not < 65.
        np.testing.assert_array_equal(out[1], [[500, 0], [0, 500]])

    # -------------------------------------------------------------------------
    # testTileList
    # -------------------------------------------------------------------------
    def testTileList(self):

        self.assertEqual(Metrics.ZENITH_ONLY_NDVI_TILES,
                         ('h15v02', 'h16v00', 'h16v01', 'h16v02',
                          'h17v00', 'h17v01'))

        self.assertNotIn('h17v02', Metrics.ZENITH_ONLY_NDVI_TILES)
