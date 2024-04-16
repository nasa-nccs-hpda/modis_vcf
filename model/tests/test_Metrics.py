
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from modis_vcf.model.Band import Band
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

        cls._outDir = Path('/explore/nobackup/people/rlgill' +      
                           '/SystemTesting/modis-vcf')

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
                               MetricsTestCase._outDir,
                               MetricsTestCase._logger)
                                   
        return self._mm
            
    # -------------------------------------------------------------------------
    # testApplyThreshold
    # -------------------------------------------------------------------------
    def testApplyThreshold(self):

        # Test the Numpy Jui Jitsu using a simple array.
        arr = np.array([[[     2, np.nan], [45, 78]], 
                        [[np.nan,     92], [60, 76]],
                        [[    76, np.nan], [20, 18]]])
        
        # Ensure the array is structured as expected.
        self.assertEqual(arr.shape, (3, 2, 2))
        
        self.assertTrue(np.array_equal(arr[:, 0, 0], 
                                       (2, np.nan, 76), 
                                       equal_nan=True))
        
        # Count the NaNs in each stack.
        self.assertEqual(np.count_nonzero(np.isnan(arr[:, 0, 0])), 1)
        self.assertEqual(np.count_nonzero(np.isnan(arr[:, 0, 1])), 2)
        self.assertEqual(np.count_nonzero(np.isnan(arr[:, 1, 0])), 0)
        self.assertEqual(np.count_nonzero(np.isnan(arr[:, 1, 1])), 0)
        
        # This is the example to encode in Metrics._applyThreshold().
        threshed = np.where(np.count_nonzero(np.isnan(arr), axis=0) > 1, 
                            np.nan, 
                            arr)
                            
        # This element should have met the threshold and modified to all NaN.
        self.assertTrue(np.isnan(threshed[:, 0, 1]).all())
        
        self.assertFalse(np.array_equal(arr[:, 0, 1], 
                                        threshed[:, 0, 1],
                                        equal_nan=True))
        
        # These elements should remain the same.
        self.assertEqual(np.count_nonzero(np.isnan(threshed[:, 0, 0])), 1)
        self.assertFalse(np.isnan(np.nanmax(threshed)))

        self.assertTrue(np.array_equal(arr[:, 0, 0], 
                                       threshed[:, 0, 0],
                                       equal_nan=True))
                                       
        self.assertTrue(np.array_equal(arr[:, 1, 0],
                                       threshed[:, 1, 0],
                                       equal_nan=True))
                                       
        self.assertTrue(np.array_equal(arr[:, 1, 1], 
                                       threshed[:, 1, 1],
                                       equal_nan=True))
                                       
        # Now test the real method.
        b5FileName = self.mm._combinedDir / (Pair.BAND5 + '.bin')

        if b5FileName.exists():
            
            os.remove(b5FileName)
            
            if Pair.BAND5 in self.mm._combinedCubes:
                del self.mm._combinedCubes[Pair.BAND5]

        self.assertFalse(b5FileName.exists())
        b5 = self.mm.getBandCube(Pair.BAND5, applyThreshold=False)

        # Use MetricsTestCase.findStacksForApplyThreshold() to discover these.
        ltIndex = (0, 0)
        eqIndex = (0, 292)
        gtIndex = (0, 298)

        # These are how many NaNs are expected at these locations.
        self.assertEqual(np.count_nonzero(np.isnan(b5[:, 0, 0])), 0)
        self.assertEqual(np.count_nonzero(np.isnan(b5[:, 0, 292])), 3)
        self.assertEqual(np.count_nonzero(np.isnan(b5[:, 0, 298])), 8)

        # Now apply the threshold.
        os.remove(b5FileName)
        del self.mm._combinedCubes[Pair.BAND5]

        b5Thresh = self.mm.getBandCube(Pair.BAND5)
        
        self.assertFalse(np.isnan(np.nanmax(b5Thresh)))
        self.assertEqual(np.count_nonzero(np.isnan(b5Thresh[:, 0, 0])), 0)
        self.assertEqual(np.count_nonzero(np.isnan(b5Thresh[:, 0, 292])), 3)
        self.assertEqual(np.count_nonzero(np.isnan(b5Thresh[:, 0, 298])), 11)

        self.assertTrue(np.array_equal(b5[:, 0, 0], 
                                       b5Thresh[:, 0, 0], 
                                       equal_nan=True))

        self.assertTrue(np.array_equal(b5[:, 0, 292], 
                                       b5Thresh[:, 0, 292], 
                                       equal_nan=True))

        self.assertFalse(np.array_equal(b5[:, 0, 299], 
                                        b5Thresh[:, 0, 298], 
                                        equal_nan=True))

    # -------------------------------------------------------------------------
    # findStacksForApplyThreshold
    # -------------------------------------------------------------------------
    def findStacksForApplyThreshold(self, band: np.ndarray):
        
        # ---
        # Find stacks to test.
        # Default threshold is 3.  Loop through pixel by pixel to identify
        # one stack with more than 3 NaNs, one stack with 3 NaNs, and one
        # stack with fewer than 3 NaNs.
        # ---
        ltThresh = None
        eqThresh = None
        gtThresh = None

        for c in range(Band.COLS):
            
            for r in range(Band.ROWS):

                stack = band[:, c, r]
                numNan = np.count_nonzero(np.isnan(stack))
                
                if numNan == self.mm._nanThreshold:
                    
                    if not eqThresh:
                        eqThresh = (c, r)
                    
                elif numNan > self.mm._nanThreshold:
                    
                    if not gtThresh:
                        gtThresh = (c, r)
                    
                elif not ltThresh:
                    
                    ltThresh = (c, r)
                    
            if ltThresh and eqThresh and gtThresh:
                break

        print('LT stack:', ltThresh)
        print('EQ stack:', eqThresh)
        print('GT stack:', gtThresh)

    # -------------------------------------------------------------------------
    # testCombine
    # -------------------------------------------------------------------------
    def testCombine(self):

        combined = self.mm.getBandCube(Pair.BAND1)
        self.assertEqual(combined.shape, (11, 4800, 4800))
        
        b1 = self.mm._cbbd.getBand(Pair.BAND1)
        x = 21
        y = 12
        m1 = b1.cube[0, x, y]
        m2 = b1.cube[1, x, y]
        mMean = (m1 + m2) / 2
        self.assertEqual(mMean, combined[0, x, y])

        m7 = b1.cube[14, x, y]
        m8 = b1.cube[15, x, y]
        mMean2 = (m7 + m8) / 2
        self.assertEqual(mMean2, combined[7, x, y])

    # -------------------------------------------------------------------------
    # testGetBandCube
    # -------------------------------------------------------------------------
    def testGetBandCube(self):

        # Test compute, write, read.
        b5FileName = self.mm._combinedDir / (Pair.BAND5 + '.bin')

        if b5FileName.exists():

            os.remove(b5FileName)
            
            if Pair.BAND5 in self.mm._combinedCubes:
                del self.mm._combinedCubes[Pair.BAND5]

        self.assertFalse(b5FileName.exists())
        b5ComputeAndWrite = self.mm.getBandCube(Pair.BAND5)
        
        xrefFile = b5FileName.with_suffix('.xref')
        self.assertTrue(xrefFile.exists())
        
        self._mm._combinedCubes = {}  # Erase b5 from cubes, so it is read.
        b5Read = self.mm.getBandCube(Pair.BAND5)
        self.assertEqual(b5ComputeAndWrite.shape, (11, Band.ROWS, Band.COLS))
        
        self.assertTrue(np.array_equal(b5ComputeAndWrite, 
                                       b5Read, 
                                       equal_nan=True))

        # Test normal call.
        b5 = self.mm.getBandCube(Pair.BAND5)
        self.assertEqual(b5.shape, (11, Band.ROWS, Band.COLS))
        raw = self.mm._cbbd.getBand(Pair.BAND5)
        x = 21
        y = 12
        p1 = raw.cube[0, x, y]
        p2 = raw.cube[1, x, y]
        exp = (p1 + p2) / 2
        self.assertEqual(exp, b5[0, x, y])
        
        # Test band 31.
        b31 = self.mm.getBandCube(Pair.BAND31)
        
    # -------------------------------------------------------------------------
    # testGetDayXref
    # -------------------------------------------------------------------------
    def testGetDayXref(self):

        # Get a combined band to ensure the xref was written.
        b5 = self.mm.getBandCube(Pair.BAND5)
        xref = self.mm.getDayXref(Pair.BAND5)
        print('xref: ' + str(xref))
        self.assertTrue('2019065' in xref)
        self.assertTrue('2020033' in xref)
        self.assertEqual(xref['2019225'], 5)
        
    # -------------------------------------------------------------------------
    # testGetNdvi
    # -------------------------------------------------------------------------
    def testGetNdvi(self):
        
        # Remove any existing versions of b1, b2 and ndvi.
        for bName in [Pair.BAND1, Pair.BAND2, Metrics.NDVI]:
        
            fName = self.mm.getCombinedFileNames(bName)[0]
            
            if fName.exists():
                os.remove(fName)
                
            if bName in self.mm._combinedCubes:
                del self.mm._combinedCubes[bName]

        b1 = self.mm.getBandCube(Pair.BAND1)
        self.assertEqual(b1.shape, (11, Band.ROWS, Band.COLS))
        self.assertTrue((b1 > 0).any())
        
        b2 = self.mm.getBandCube(Pair.BAND2)
        self.assertEqual(b2.shape, (11, Band.ROWS, Band.COLS))
        self.assertTrue((b2 > 0).any())
        
        # ---
        # Compute without threshold.
        #
        # From running findStacksForApplyThreshold(), we know that 
        # [:, 0, 0] is below the threshold
        # [:, 0, 292] is at the threshold
        # [:, 0, 298] is above the threshold
        #
        # Therefore, after application of the threshold:
        # [:, 0, 0] should be the same
        # [:, 0, 292] should be the same
        # [:, 0, 298] should be all NaN
        # ---
        ndviNoThr = self.mm.getNdvi(applyThreshold=False)
        self.findStacksForApplyThreshold(ndviNoThr)
        self.assertEqual(np.count_nonzero(np.isnan(ndviNoThr[:, 0, 0])), 0)
        self.assertEqual(np.count_nonzero(np.isnan(ndviNoThr[:, 0, 292])), 3)
        self.assertEqual(np.count_nonzero(np.isnan(ndviNoThr[:, 0, 298])), 11)
        
        ndviThr = self.mm._applyThreshold(ndviNoThr)
        self.assertEqual(np.count_nonzero(np.isnan(ndviThr[:, 0, 0])), 0)
        self.assertEqual(np.count_nonzero(np.isnan(ndviThr[:, 0, 292])), 3)
        self.assertEqual(np.count_nonzero(np.isnan(ndviThr[:, 0, 298])), 11)
        
        # Apply the threshold within getNdvi().
        fName = self.mm.getCombinedFileNames(Metrics.NDVI)[0]
        os.remove(fName)
        self.mm._ndvi = None
        ndvi = self.mm.getNdvi()
        self.assertEqual(ndvi.shape, (11, Band.ROWS, Band.COLS))
        self.assertTrue((ndvi > 0).any())  # not all 0, like in training file
        self.assertEqual(np.count_nonzero(np.isnan(ndvi[:, 0, 0])), 0)
        self.assertEqual(np.count_nonzero(np.isnan(ndvi[:, 0, 292])), 3)
        self.assertEqual(np.count_nonzero(np.isnan(ndvi[:, 0, 298])), 11)

        # Test reading ndvi.
        ndviRead = self.mm.getNdvi()
        self.assertEqual(self.mm.getNdvi().shape, (11, Band.ROWS, Band.COLS))
        self.assertTrue(np.array_equal(ndvi, ndviRead, equal_nan=True))
        
    # -------------------------------------------------------------------------
    # testRegistration
    # -------------------------------------------------------------------------
    def testRegistration(self):
        
        print('Available metrics:', self.mm.availableMetrics)
        self.mm.printAvailableMetrics()
        
    # -------------------------------------------------------------------------
    # testAmpBandReflZeros
    # -------------------------------------------------------------------------
    def testAmpBandReflZeros(self):
        
        METRIC_NAME = 'metricAmpBandRefl'
        metFileName = self.mm._getOutName(METRIC_NAME)
        
        if metFileName.exists():
            os.remove(metFileName)
            
        abr = self.mm.getMetric(METRIC_NAME)
        self.assertEqual(abr.cube[0, 0, 298], Band.NO_DATA)
        self.assertEqual(abr.cube[0, 0, 303], Band.NO_DATA)
        
    # -------------------------------------------------------------------------
    # testBandReflMedianGreenness
    # -------------------------------------------------------------------------
    def testBandReflMedianGreenness(self):
        
        METRIC_NAME = 'metricBandReflMedianGreenness'
        metFileName = self.mm._getOutName(METRIC_NAME)
        
        if metFileName.exists():
            os.remove(metFileName)
            
        metric = self.mm.getMetric(METRIC_NAME)
        print(metric.cube[:, 0, 531])
        
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

        