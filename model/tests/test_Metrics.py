
import logging
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.Pair import Pair
from modis_vcf.model.Utils import Utils


# -----------------------------------------------------------------------------
# class MetricsTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_Metrics
# python -m unittest modis_vcf.model.tests.test_Metrics.MetricsTestCase.testGetSortKeys
# -----------------------------------------------------------------------------
class MetricsTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUpClass
    # -------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):

        # Define valid parameters.
        cls._validTileId = 'h09v05'
        cls._validYear = 2019
        
        cls._realInDir = \
            Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')

        cls._logger = logging.getLogger()
        cls._logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        cls._logger.addHandler(ch)

    # -------------------------------------------------------------------------
    # setUp
    # -------------------------------------------------------------------------
    def setUp(self):

        self._mm = None
        
    # -------------------------------------------------------------------------
    # mm
    # -------------------------------------------------------------------------
    @property
    def mm(self):
        
        if not self._mm:
            
            self._mm = Metrics(MetricsTestCase._validTileId,
                                   MetricsTestCase._validYear,
                                   MetricsTestCase._realInDir,
                                   MetricsTestCase._logger)
                                   
        return self._mm
            
    # -------------------------------------------------------------------------
    # testCombine
    # -------------------------------------------------------------------------
    def testCombine(self):
        
        # The constructor calls _combine().
        combined = self.mm.getBand(Pair.BAND1)
        self.assertEqual(combined.cube.shape, (12, 4800, 4800))
        
        b1 = self.mm._cbbd.getBand(Pair.BAND1)
        x = 21
        y = 12
        m1 = b1.cube[0, x, y]
        m2 = b1.cube[1, x, y]
        mMean = (m1 + m2) / 2
        self.assertEqual(mMean, combined.cube[0, x, y])

        m7 = b1.cube[14, x, y]
        m8 = b1.cube[15, x, y]
        mMean2 = (m7 + m8) / 2
        self.assertEqual(mMean2, combined.cube[7, x, y])

    # -------------------------------------------------------------------------
    # testGetBand
    # -------------------------------------------------------------------------
    def testGetBand(self):
        
        b5 = self.mm.getBand(Pair.BAND5)
        raw = self.mm._cbbd.getBand(Pair.BAND5)
        x = 21
        y = 12
        p1 = raw.cube[0, x, y]
        p2 = raw.cube[1, x, y]
        exp = (p1 + p2) / 2
        self.assertEqual(exp, b5.cube[0, x, y])

    # -------------------------------------------------------------------------
    # testNdvi
    # -------------------------------------------------------------------------
    def testNdvi(self):
        
        self.mm.ndvi
        self.mm.ndvi
        
    # -------------------------------------------------------------------------
    # testRegistration
    # -------------------------------------------------------------------------
    def testRegistration(self):
        
        print('Available metrics:', self.mm.availableMetrics())
        self.mm.printAvailableMetrics()
        
    # -------------------------------------------------------------------------
    # testSort
    # -------------------------------------------------------------------------
    # def testSort(self):
    #
    #     mm = self.mm
    #     b1 = mm.getBand(Pair.BAND1)
    #     sortedBand = mm._sort(b1)
    #
    #     self.assertIsInstance(sortedBand, Utils.Band)
    #     self.assertTrue((sortedBand.cube == b1.cube).all())
    #
    #     # Test NDVI sort.
    #     ndviSortedBand = mm._sort(b1, Metrics.SortMethod.NDVI)
    #     self.assertFalse((ndviSortedBand.cube == b1.cube).all())
    #
    #     # Find the largest NDVI value at the location.
    #     x = 21
    #     y = 12
    #     maxIndex = mm.ndvi.cube[:, x, y].argmax()
    #     self.assertEqual(ndviSortedBand.cube[0, x, y], b1.cube[maxIndex, x, y])
    #
    #     # Test thermal sort.
    #     thermSortedBand = mm._sort(b1, Metrics.SortMethod.THERMAL)
    #     self.assertFalse((thermSortedBand.cube == b1.cube).all())
    #
    #     # Find the largest thermal value at the location.
    #     x = 2100
    #     y = 1200
    #     raw31 = mm._cbbd.getBand(Pair.BAND31)
    #     combined31 = mm._combine(raw31)
    #     maxIndex = combined31.cube[:, x, y].argmax()
    #
    #     self.assertEqual(thermSortedBand.cube[0, x, y],
    #                      b1.cube[maxIndex, x, y])

    # -------------------------------------------------------------------------
    # testWriteMetrics
    # -------------------------------------------------------------------------
    # def testWriteMetrics(self):
    #
    #     b1 = self.mm.getBand(Pair.BAND1)
    #     b2 = self.mm.getBand(Pair.BAND2)
    #     self.mm.metric1(b1)
    #     self.mm.metric1(b2)
    #     self.mm.metric4(b1)
    #     self.mm.metric4(b2)
    #     outDir = Path(tempfile.mkdtemp())
    #     self.mm.writeMetrics(outDir)

    # # -------------------------------------------------------------------------
    # # testMetric1
    # # -------------------------------------------------------------------------
    # def testMetric1(self):
    #
    #     b1 = self.mm.getBand(Pair.BAND1)
    #     result = self.mm.metric1(b1)
    #     outDir = Path(tempfile.mkdtemp())
    #     self.mm.writeDebugMetrics(outDir, b1)
    #     self.mm.writeMetrics(outDir)
    #     self.assertEqual(-11791.5, result[21, 120])
    #
    # # -------------------------------------------------------------------------
    # # testMetric4
    # #
    # # Band1-raw bands are Int16
    # # HDF4_EOS:EOS_GRID:"/explore/nobackup/projects/ilab/data/MODIS/MOD44C/MOD44CQ.A2019065.h09v05.061.2020290183523.hdf":MOD44C_250m_GRID:Band_1
    # # Band 1 Block=4800x4800 Type=Int16, ColorInterp=Gray
    # # Description = Band 1 Reflectance
    # # Min=-100.000 Max=13203.000
    # # Minimum=-100.000, Maximum=13203.000, Mean=1808.514, StdDev=1615.520
    # # NoData Value=36864
    # # -------------------------------------------------------------------------
    # def testMetric4(self):
    #
    #     b1 = self.mm.getBand(Pair.BAND1)
    #     result = self.mm.metric4(b1)
    #     outDir = Path(tempfile.mkdtemp())
    #     self.mm.writeDebugMetrics(outDir, b1)
    #     self.mm.writeMetrics(outDir)
    #     self.assertEqual(-9635, result[21, 120])
    #
    # # -------------------------------------------------------------------------
    # # testMetric13
    # # -------------------------------------------------------------------------
    # def testMetric13(self):
    #
    #     b2 = self.mm.getBand(Pair.BAND2)
    #     result = self.mm.metric13(b2)
    #     self.assertEqual(-2048, result[21, 12])
    #
    # # -------------------------------------------------------------------------
    # # testMetric27
    # # -------------------------------------------------------------------------
    # def testMetric27(self):
    #
    #     b3 = self.mm.getBand(Pair.BAND3)
    #     result = self.mm.metric27(b3)
    #     self.assertEqual(-1994.5, result[210, 120])
        