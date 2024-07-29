
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from modis_vcf.model.Band import Band
from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.ProductTypeMod09A import ProductTypeMod09A
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44


# -----------------------------------------------------------------------------
# class MetricsTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_Metrics
# python -m unittest modis_vcf.model.tests.test_Metrics.MetricsTestCase.testMod09
# -----------------------------------------------------------------------------
class MetricsTestCase(unittest.TestCase):

    NUM_MOD44_SPLITS = 12
    
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
                           '/SystemTesting/modis-vcf/MOD44')

        cls._logger = logging.getLogger()
        cls._logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        cls._logger.addHandler(ch)
        cls._productType = ProductTypeMod44(cls._realInDir)

        cls.days = [(2019,  65), (2019,  97), (2019, 129), (2019, 161),
                    (2019, 193), (2019, 225), (2019, 257), (2019, 289),
                    (2019, 321), (2019, 353), (2020,  17), (2020, 49)]

    # -------------------------------------------------------------------------
    # setUp
    # -------------------------------------------------------------------------
    def setUp(self):

        self._mm = None
        
        baseDir = MetricsTestCase._outDir / \
                  (MetricsTestCase._validTileId + \
                  '-' + \
                   str(MetricsTestCase._validYear))
        
        dayDir = baseDir / '1-Days'
        compDir = baseDir / '2-Composites'
        metricsDir = baseDir / '3-Metrics'
        
        print('baseDir:', baseDir)
        print('dayDir:', dayDir)
        print('compDir:', compDir)
        print('metricsDir:', metricsDir)

        # files = dayDir.glob('*.bin')
        # for f in files: f.unlink()
        #
        # files = compDir.glob('*.bin')
        # for f in files: f.unlink()
        #
        # files = metricsDir.glob('*.tif')
        # for f in files: f.unlink()
        
    # -------------------------------------------------------------------------
    # mm
    # -------------------------------------------------------------------------
    @property
    def mm(self):
        
        if not self._mm:
            
            self._mm = Metrics(MetricsTestCase._validTileId,
                               MetricsTestCase._validYear,
                               MetricsTestCase._productType,
                               # MetricsTestCase._realInDir,
                               MetricsTestCase._outDir,
                               MetricsTestCase._logger)
                                   
        return self._mm
            
    # -------------------------------------------------------------------------
    # testApplyThreshold
    # -------------------------------------------------------------------------
    # def testApplyThreshold(self):
    #
    #     # Test the Numpy Jui Jitsu using a simple array.
    #     arr = np.array([[[     2, np.nan], [45, 78]],
    #                     [[np.nan,     92], [60, 76]],
    #                     [[    76, np.nan], [20, 18]]])
    #
    #     # Ensure the array is structured as expected.
    #     self.assertEqual(arr.shape, (3, 2, 2))
    #
    #     self.assertTrue(np.array_equal(arr[:, 0, 0],
    #                                    (2, np.nan, 76),
    #                                    equal_nan=True))
    #
    #     # Count the NaNs in each stack.
    #     self.assertEqual(np.count_nonzero(np.isnan(arr[:, 0, 0])), 1)
    #     self.assertEqual(np.count_nonzero(np.isnan(arr[:, 0, 1])), 2)
    #     self.assertEqual(np.count_nonzero(np.isnan(arr[:, 1, 0])), 0)
    #     self.assertEqual(np.count_nonzero(np.isnan(arr[:, 1, 1])), 0)
    #
    #     # This is the example to encode in Metrics._applyThreshold().
    #     threshed = np.where(np.count_nonzero(np.isnan(arr), axis=0) > 1,
    #                         np.nan,
    #                         arr)
    #
    #     # This element should have met the threshold and modified to all NaN.
    #     self.assertTrue(np.isnan(threshed[:, 0, 1]).all())
    #
    #     self.assertFalse(np.array_equal(arr[:, 0, 1],
    #                                     threshed[:, 0, 1],
    #                                     equal_nan=True))
    #
    #     # These elements should remain the same.
    #     self.assertEqual(np.count_nonzero(np.isnan(threshed[:, 0, 0])), 1)
    #     self.assertFalse(np.isnan(np.nanmax(threshed)))
    #
    #     self.assertTrue(np.array_equal(arr[:, 0, 0],
    #                                    threshed[:, 0, 0],
    #                                    equal_nan=True))
    #
    #     self.assertTrue(np.array_equal(arr[:, 1, 0],
    #                                    threshed[:, 1, 0],
    #                                    equal_nan=True))
    #
    #     self.assertTrue(np.array_equal(arr[:, 1, 1],
    #                                    threshed[:, 1, 1],
    #                                    equal_nan=True))
    #
    #     # Now test the real method.
    #     b5FileName = self.mm._combinedDir / (self.mm._productType.BAND5 + '.bin')
    #
    #     if b5FileName.exists():
    #
    #         os.remove(b5FileName)
    #
    #         if self.mm._productType.BAND5 in self.mm._combinedCubes:
    #             del self.mm._combinedCubes[self.mm._productType.BAND5]
    #
    #     self.assertFalse(b5FileName.exists())
    #     b5 = self.mm.getBandCube(self.mm._productType.BAND5, applyThreshold=False)
    #
    #     # Use MetricsTestCase.findStacksForApplyThreshold() to discover these.
    #     ltIndex = (0, 0)
    #     eqIndex = (0, 292)
    #     gtIndex = (0, 298)
    #
    #     # These are how many NaNs are expected at these locations.
    #     self.assertEqual(np.count_nonzero(np.isnan(b5[:, 0, 0])), 0)
    #     self.assertEqual(np.count_nonzero(np.isnan(b5[:, 0, 292])), 3)
    #     self.assertEqual(np.count_nonzero(np.isnan(b5[:, 0, 298])), 8)
    #
    #     # Now apply the threshold.
    #     os.remove(b5FileName)
    #     del self.mm._combinedCubes[self.mm._productType.BAND5]
    #
    #     b5Thresh = self.mm.getBandCube(self.mm._productType.BAND5)
    #
    #     self.assertFalse(np.isnan(np.nanmax(b5Thresh)))
    #     self.assertEqual(np.count_nonzero(np.isnan(b5Thresh[:, 0, 0])), 0)
    #     self.assertEqual(np.count_nonzero(np.isnan(b5Thresh[:, 0, 292])), 3)
    #
    #     self.assertEqual(np.count_nonzero(np.isnan(b5Thresh[:, 0, 298])),
    #                      MetricsTestCase.NUM_MOD44_SPLITS)
    #
    #     self.assertTrue(np.array_equal(b5[:, 0, 0],
    #                                    b5Thresh[:, 0, 0],
    #                                    equal_nan=True))
    #
    #     self.assertTrue(np.array_equal(b5[:, 0, 292],
    #                                    b5Thresh[:, 0, 292],
    #                                    equal_nan=True))
    #
    #     self.assertFalse(np.array_equal(b5[:, 0, 299],
    #                                     b5Thresh[:, 0, 298],
    #                                     equal_nan=True))

    # -------------------------------------------------------------------------
    # testApplyThreshold
    # 
    # V2 test successful
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
        b5Name = self.mm._productType.BAND5
        b5, b5Xref = self.mm.getBandCube(b5Name, applyThreshold=False)

        # Use MetricsTestCase.findStacksForApplyThreshold() to discover these.
        # self.findStacksForApplyThreshold(b5)
        ltx = 0
        lty = 0
        eqx = 0
        eqy = 232
        gtx = 0
        gty = 292

        # These are how many NaNs are expected at these locations.
        self.assertEqual(np.count_nonzero(np.isnan(b5[:, ltx, lty])), 0)

        testDays = []
        
        for day in MetricsTestCase.days:
            
            name = 'MOD44-' + str(day[0]) + str(day[1]).zfill(3) + \
                   '-' + b5Name + '.bin'
                   
            fName = Path(self.mm._outDir / '1-Days') / name
            raster = np.fromfile(fName)
            testDays.append(raster)

        self.assertEqual(np.count_nonzero(np.isnan(b5[:, eqx, eqy])), 3)
        self.assertEqual(np.count_nonzero(np.isnan(b5[:, gtx, gty])), 4)

        # Now apply the threshold.
        files = self.mm._metricsDir.glob('*.tif')
        for f in files: f.unlink()
        self.mm._metricsDir.rmdir()
        b5Thresh, b5Xref = self.mm.getBandCube(b5Name)
        
        self.assertFalse(np.isnan(np.nanmax(b5Thresh)))
        self.assertEqual(np.count_nonzero(np.isnan(b5Thresh[:, ltx, lty])), 0)
        self.assertEqual(np.count_nonzero(np.isnan(b5Thresh[:, eqx, eqy])), 3)
        
        self.assertEqual(np.count_nonzero(np.isnan(b5Thresh[:, gtx, gty])),
                         MetricsTestCase.NUM_MOD44_SPLITS)

        self.assertTrue(np.array_equal(b5[:, ltx, lty], 
                                       b5Thresh[:, ltx, lty], 
                                       equal_nan=True))

        self.assertTrue(np.array_equal(b5[:, eqx, eqy], 
                                       b5Thresh[:, eqx, eqy], 
                                       equal_nan=True))

        self.assertFalse(np.array_equal(b5[:, gtx, gty], 
                                        b5Thresh[:, gtx, gty], 
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
                    
                elif numNan > self.mm._nanThreshold and numNan < 12:
                    
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
    # def testCombine(self):
    #
    #     combined = self.mm.getBandCube(self.mm._productType.BAND1)
    #
    #     self.assertEqual(combined.shape,
    #                      (MetricsTestCase.NUM_MOD44_SPLITS, 4800, 4800))
    #
    #     b1 = self.mm._cbbd.getBand(self.mm._productType.BAND1)
    #     x = 21
    #     y = 12
    #     m1 = b1.cube[0, x, y]
    #     m2 = b1.cube[1, x, y]
    #     mMean = (m1 + m2) / 2
    #     self.assertEqual(mMean, combined[0, x, y])
    #
    #     m7 = b1.cube[14, x, y]
    #     m8 = b1.cube[15, x, y]
    #     mMean2 = (m7 + m8) / 2
    #     self.assertEqual(mMean2, combined[7, x, y])

    # -------------------------------------------------------------------------
    # testGetBandCube
    # -------------------------------------------------------------------------
    # def testGetBandCube(self):
    #
    #     # Test compute, write, read.
    #     b5FileName = self.mm._combinedDir / (self.mm._productType.BAND5 + '.bin')
    #
    #     if b5FileName.exists():
    #
    #         os.remove(b5FileName)
    #
    #         if self.mm._productType.BAND5 in self.mm._combinedCubes:
    #             del self.mm._combinedCubes[self.mm._productType.BAND5]
    #
    #     self.assertFalse(b5FileName.exists())
    #     b5ComputeAndWrite = self.mm.getBandCube(self.mm._productType.BAND5)
    #
    #     xrefFile = b5FileName.with_suffix('.xref')
    #     self.assertTrue(xrefFile.exists())
    #
    #     self._mm._combinedCubes = {}  # Erase b5 from cubes, so it is read.
    #     b5Read = self.mm.getBandCube(self.mm._productType.BAND5)
    #
    #     self.assertEqual(b5ComputeAndWrite.shape,
    #                     (MetricsTestCase.NUM_MOD44_SPLITS,
    #                      Band.ROWS,
    #                      Band.COLS))
    #
    #     self.assertTrue(np.array_equal(b5ComputeAndWrite,
    #                                    b5Read,
    #                                    equal_nan=True))
    #
    #     # Test normal call.
    #     b5 = self.mm.getBandCube(self.mm._productType.BAND5)
    #
    #     self.assertEqual(b5.shape,
    #                     (MetricsTestCase.NUM_MOD44_SPLITS,
    #                      Band.ROWS,
    #                      Band.COLS))
    #
    #     raw = self.mm._cbbd.getBand(self.mm._productType.BAND5)
    #     x = 21
    #     y = 12
    #     p1 = raw.cube[0, x, y]
    #     p2 = raw.cube[1, x, y]
    #     exp = (p1 + p2) / 2
    #     self.assertEqual(exp, b5[0, x, y])
    #
    #     # Test band 31.
    #     b31 = self.mm.getBandCube(self.mm._productType.BAND31)
        
    # -------------------------------------------------------------------------
    # testGetBandCube
    #
    # V2 test successful
    # -------------------------------------------------------------------------
    def testGetBandCube(self):

        b5, b5Xref = self.mm.getBandCube(self.mm._productType.BAND5)
        self.assertEqual(b5.shape, (12, 4800, 4800))
        self.assertEqual(b5.dtype, np.float64)

        days = ['2019065', '2019097', '2019129', '2019161', '2019193',
                '2019225', '2019257', '2019289', '2019321', '2019353',
                '2020017', '2020049', ]

        self.assertEqual(list(b5Xref.keys()), days)
        
    # -------------------------------------------------------------------------
    # testCompositeValues
    #
    # V2 test failure
    # -------------------------------------------------------------------------
    def testCompositeValues(self):

        b5Name = self.mm._productType.BAND5
        b5, b5Xref = self.mm.getBandCube(b5Name, False)
        self.assertEqual(np.count_nonzero(np.isnan(b5[:, 0, 0])), 0)

        daysSought = [(2019, 65), (2019, 97), (2019, 129), (2019, 161), 
                      (2019, 193), (2019, 225), (2019, 257), (2019, 289),
                      (2019, 321), (2019, 353), (2020, 17), (2020, 49)]
        
        testDays = []
        
        for year, day in daysSought:
            
            name = 'MOD44-' + str(year) + str(day).zfill(3) + \
                   '-' + b5Name + '.bin'
                   
            fName = Path(self.mm._outDir / '2-Composites') / name
            raster = np.fromfile(fName, dtype=np.float64).reshape(4800, 4800)
            testDays.append(raster)

        for i in range(12):
            self.assertEqual(b5[i, 0, 0], testDays[i][0, 0])

        self.assertEqual(np.count_nonzero(np.isnan(b5[:, 0, 232])), 3)
        self.assertEqual(np.count_nonzero(np.isnan(b5[:, 0, 292])), 4)

    # -------------------------------------------------------------------------
    # testGetDayXref
    #
    # V2 test successful
    # -------------------------------------------------------------------------
    def testGetDayXref(self):

        # Get a combined band to ensure the xref was written.
        b5, xref = self.mm.getBandCube(self.mm._productType.BAND5)
        # xref = self.mm.getDayXref(self.mm._productType.BAND5)
        print('xref: ' + str(xref))
        self.assertTrue('2019065' in xref)
        self.assertTrue('2020017' in xref)
        self.assertEqual(xref['2019225'], 5)
        
    # -------------------------------------------------------------------------
    # testGetNdvi
    #
    # V2 test successful
    # -------------------------------------------------------------------------
    def testGetNdvi(self):
        
        # Remove any existing versions of b1, b2 and ndvi.
        # for bName in [self.mm._productType.BAND1,
        #               self.mm._productType.BAND2,
        #               Metrics.NDVI]:
        #
        #     # fName = self.mm.getCombinedFileNames(bName)[0]
        #
        #     if fName.exists():
        #         os.remove(fName)
        #
        #     if bName in self.mm._combinedCubes:
        #         del self.mm._combinedCubes[bName]

        b1, b1Xref = self.mm.getBandCube(self.mm._productType.BAND1)
        
        self.assertEqual(b1.shape, 
                        (MetricsTestCase.NUM_MOD44_SPLITS, 
                         self.mm._productType.ROWS, 
                         self.mm._productType.COLS))
                         
        self.assertTrue((b1 > 0).any())
        
        b2, b2Xref = self.mm.getBandCube(self.mm._productType.BAND2)
        
        self.assertEqual(b2.shape, 
                        (MetricsTestCase.NUM_MOD44_SPLITS, 
                         self.mm._productType.ROWS, 
                         self.mm._productType.COLS))
                         
        self.assertTrue((b2 > 0).any())
        
        # ---
        # Compute without threshold.
        #
        # From running findStacksForApplyThreshold(), we know that 
        # [:, 0, 0] is below the threshold
        # [:, 0, 232] is at the threshold
        # [:, 4242, 4494] is above the threshold
        #
        # Therefore, after application of the threshold:
        # [:, 0, 0] should be the same
        # [:, 0, 232] should be the same
        # [:, 4242, 4494] should be all NaN
        # ---
        ndviNoThr, ndviXref = self.mm.getNdvi(applyThreshold=False)
        # self.findStacksForApplyThreshold(ndviNoThr)
        self.assertEqual(np.count_nonzero(np.isnan(ndviNoThr[:, 0, 0])), 0)
        self.assertEqual(np.count_nonzero(np.isnan(ndviNoThr[:, 0, 232])), 3)
        
        self.assertEqual( \
            np.count_nonzero(np.isnan(ndviNoThr[:, 4242, 4494])), 4)
        
        ndviThr = self.mm._applyThreshold(ndviNoThr)
        self.assertEqual(np.count_nonzero(np.isnan(ndviThr[:, 0, 0])), 0)
        self.assertEqual(np.count_nonzero(np.isnan(ndviThr[:, 0, 232])), 3)
        
        self.assertEqual( \
            np.count_nonzero(np.isnan(ndviThr[:, 4242, 4494])), 12)
        
        # Apply the threshold within getNdvi().
        self.mm._ndvi = None
        ndvi, ndviXref = self.mm.getNdvi()
        
        self.assertEqual(ndvi.shape, (12, self.mm._productType.ROWS,
                                          self.mm._productType.COLS))
        
        self.assertTrue((ndvi > 0).any())  # not all 0, like in training file
        self.assertEqual(np.count_nonzero(np.isnan(ndvi[:, 0, 0])), 0)
        self.assertEqual(np.count_nonzero(np.isnan(ndvi[:, 0, 232])), 3)
        self.assertEqual(np.count_nonzero(np.isnan(ndvi[:, 4242, 4494])), 12)

        # Test reading ndvi.
        ndviRead, ndviXref = self.mm.getNdvi()
        
        self.assertEqual(self.mm.getNdvi()[0].shape,
                         (MetricsTestCase.NUM_MOD44_SPLITS, 
                         self.mm._productType.ROWS, 
                         self.mm._productType.COLS))
                         
        self.assertTrue(np.array_equal(ndvi, ndviRead, equal_nan=True))
        
    # -------------------------------------------------------------------------
    # testGetNumSplits
    #
    # V2 obsolete
    # -------------------------------------------------------------------------
    # def testGetNumSplits(self):
    #
    #     numSplits, allDays = self.mm._getNumSplits()
    #     self.assertEqual(numSplits, MetricsTestCase.NUM_MOD44_SPLITS)
    #     self.assertEqual(len(allDays), 23)
        
    # -------------------------------------------------------------------------
    # testRegistration
    # 
    # V2 test successful
    # -------------------------------------------------------------------------
    def testRegistration(self):
        
        print('Available metrics:', self.mm.availableMetrics)
        self.mm.printAvailableMetrics()
        
    # -------------------------------------------------------------------------
    # testSort
    # 
    # V2 test successful
    # -------------------------------------------------------------------------
    def testSort(self):

        mm = self.mm
        b1, b1Xref = mm.getBandCube(self.mm._productType.BAND1)
        sortedBand: np.ndarray = np.sort(b1, axis=0)
        self.assertFalse((sortedBand == b1).all())

        # Test NDVI sort.
        ndviSortedBand: np.ndarray = mm._sortByNDVI(b1)
        self.assertFalse((ndviSortedBand == b1).all())

        # Find the largest NDVI value at the location.
        x = 21
        y = 12
        ndvi, ndviXref = mm.getNdvi()
        maxIndex = ndvi[:, x, y].argmax()
        self.assertEqual(maxIndex, 1)
        self.assertAlmostEqual(ndvi[maxIndex, x, y], 364.26116838)
        self.assertEqual(ndviSortedBand[-1, x, y], b1[maxIndex, x, y])

        # Test thermal sort.
        thermSortedBand: np.ndarray = mm._sortByThermal(b1)
        self.assertFalse((thermSortedBand == b1).all())

        # Find the largest thermal value at the location.
        x = 2100
        y = 1200
        b31 = mm.getBandCube(self.mm._productType.BAND31)[0]
        maxIndex = b31[:, x, y].argmax()
        self.assertEqual(thermSortedBand[-1, x, y], b1[maxIndex, x, y])
        
        # Find a NaN case.
        x, y = np.argwhere(np.isnan(ndvi[11, :, :]))[0]
        self.assertTrue(np.isnan(ndvi[11, x, y]))

        minIndex = np.nanargmin(ndvi[:, x, y])
        self.assertEqual(minIndex, 0)
        self.assertAlmostEqual(ndvi[minIndex, x, y], 206.72921385)
        self.assertEqual(ndviSortedBand[0, x, y], b1[minIndex, x, y])

        maxIndex = np.nanargmax(ndvi[:, x, y])
        self.assertEqual(maxIndex, 2)
        self.assertAlmostEqual(ndvi[maxIndex, x, y], 337.47537505)
        numNan = (np.isnan(ndvi[:, x, y])).sum()
        
        self.assertEqual(ndviSortedBand[-(numNan + 1), x, y], 
                         b1[maxIndex, x, y])
        
    # -------------------------------------------------------------------------
    # testAmpBandReflZeros
    #
    # V2 test successsful
    # -------------------------------------------------------------------------
    def testAmpBandReflZeros(self):
        
        METRIC_NAME = 'metricAmpBandRefl'
        metFileName = self.mm._getOutName(METRIC_NAME)
        
        if metFileName.exists():
            os.remove(metFileName)
            
        abr = self.mm.getMetric(METRIC_NAME)
        self.assertEqual(abr.cube[0, 0, 298], self.mm._productType.NO_DATA)
        self.assertEqual(abr.cube[0, 0, 303], self.mm._productType.NO_DATA)
        
    # -------------------------------------------------------------------------
    # testBandReflMedianGreenness
    # -------------------------------------------------------------------------
    def testBandReflMedianGreenness(self):
        
        METRIC_NAME = 'metricBandReflMedianGreenness'
        metFileName = self.mm._getOutName(METRIC_NAME)
        
        mm = self.mm
        metrics = mm.metricBandReflMedianGreenness()
        self.assertEqual(len(metrics), 8)
        self.assertEqual(metrics[0].value.shape, (4800, 4800))
        self.assertEqual(metrics[0].name, 'BandReflMedianGreenness-Band_1')

        ndvi, nXref = mm.getNdvi()
        nSorted = np.sort(ndvi, axis=0)
        b1, xref = mm.getBandCube(self.mm._productType.BAND1)
        bSorted = np.sort(b1, axis=0)
        bSortedByN = self.mm._sortByNDVI(b1)
        
        # Case 1:  band with no NaNs and NDVI with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(nSorted[11, :, :]))[0]
        
        numNdviNotNan = (~np.isnan(ndvi[:, x, y])).sum(axis=0)
        index = int(numNdviNotNan / 2)
        exp = int(bSortedByN[index, x, y])
        self.assertEqual(metrics[0].value[x, y], exp)
        
        # ---
        # Case 2:  band with no NaNs and NDVI with some NaNs
        # ---
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           np.isnan(nSorted[11, :, :]))[0]
                           
        numNdviNotNan = (~np.isnan(ndvi[:, x, y])).sum(axis=0)
        index = int(numNdviNotNan / 2)  # 11/2 = 5.5 ==> 5
        exp = int(bSortedByN[index, x, y])
        self.assertEqual(metrics[0].value[x, y], exp)
                           
        # ---
        # Case 3:  band with some NaNs and NDVI with no NaNs
        # Unable to find this case.
        # x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
        #                    ~np.isnan(bSorted[1, :, :]) &
        #                    np.isnan(bSorted[11, :, :]) &
        #                    ~np.isnan(nSorted[11, :, :]))[0]
        # ---
                           
        # Case 4:  band with some NaNs and NDVI with some NaNs
        x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
                           ~np.isnan(bSorted[1, :, :]) &
                           np.isnan(bSorted[11, :, :]) &
                           np.isnan(nSorted[11, :, :]))[0]
        
        numNdviNotNan = (~np.isnan(ndvi[:, x, y])).sum(axis=0)
        index = int(numNdviNotNan / 2)
        exp = int(bSortedByN[index, x, y])
        self.assertEqual(metrics[0].value[x, y], exp)
        
        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertEqual(metrics[0].value[x, y], mm._productType.NO_DATA)
        
        # Case 6:  thermal all NaN
        x, y = np.argwhere(np.isnan(nSorted[0, :, :]))[0]
        self.assertEqual(metrics[0].value[x, y], mm._productType.NO_DATA)
        
    # -------------------------------------------------------------------------
    # testMetricAmpGreenestBandRefl
    #
    # V2 test successsful
    # -------------------------------------------------------------------------
    def testMetricAmpGreenestBandRefl(self):

        x = 100
        y = 1001
        mm = self.mm
        b1, b1Xref = mm.getBandCube(self.mm._productType.BAND1)
        ndvi, ndviXref = mm.getNdvi()
        
        maxBand = np.nanargmax(ndvi[:, x, y])
        minBand = np.nanargmin(ndvi[:, x, y])
        self.assertEqual(maxBand, 2)
        self.assertEqual(minBand, 0)
        
        b1MaxB = b1[maxBand, x, y]
        b1MinB = b1[minBand, x, y]
        self.assertEqual(b1MaxB, 1550.0)
        self.assertEqual(b1MinB, 1256.0)

        metric = mm.metricAmpGreenestBandRefl()
        b1Metric = metric[0]
        self.assertTrue(b1Metric.name, 'AmpGreenestBandRefl-Band_1')
        self.assertEqual(b1Metric.value[x, y], b1MaxB - b1MinB)

        # Find a nan case.
        x, y = np.argwhere(np.isnan(ndvi[11, :, :]))[0]
        numNan = (np.isnan(ndvi[:, x, y])).sum()
        maxBand = np.nanargmax(ndvi[:, x, y])
        minBand = np.nanargmin(ndvi[:, x, y])
        b1MaxB = b1[maxBand, x, y]
        b1MinB = b1[minBand, x, y]
        ndviSortedBand: np.ndarray = mm._sortByNDVI(b1)
        sortedBandXy = ndviSortedBand[:, x, y]
        amp = abs(b1MaxB - b1MinB)
        self.assertTrue(np.isnan(sortedBandXy[-numNan]))
        self.assertEqual(amp, abs(b1MaxB - b1MinB))
        self.assertEqual(b1Metric.value[x, y], abs(b1MaxB - b1MinB))

    # -------------------------------------------------------------------------
    # testMetricAmpWarmestBandRefl
    #
    # V2 test successsful
    # -------------------------------------------------------------------------
    def testMetricAmpWarmestBandRefl(self):

        x = 100
        y = 1001
        mm = self.mm
        b1, b1Xref = mm.getBandCube(self.mm._productType.BAND1)
        thermal, tXref = mm.getBandCube(self.mm._productType.BAND31)

        maxBand = np.nanargmax(thermal[:, x, y])
        minBand = np.nanargmin(thermal[:, x, y])
        self.assertEqual(maxBand, 4)
        self.assertEqual(minBand, 9)

        b1MaxB = b1[maxBand, x, y]
        b1MinB = b1[minBand, x, y]
        self.assertEqual(b1MaxB, 1575.0)
        self.assertEqual(b1MinB, 1302.0)

        sortedBand: np.ndarray = mm._sortByThermal(b1)
        sortedBandXy = sortedBand[:, x, y]
        amp = abs(b1MaxB - b1MinB)
        self.assertEqual(b1MaxB, sortedBandXy[-1])
        self.assertEqual(b1MinB, sortedBandXy[0])
        self.assertEqual(amp, abs(b1MaxB - b1MinB))

        metric = mm.metricAmpWarmestBandRefl()
        b1Metric = metric[0]
        self.assertTrue(b1Metric.name, 'AmpWarmestBandRefl-Band_1')
        self.assertEqual(b1Metric.value[x, y], int(amp))

        # Find a nan case.  Ensure that it is not all NaN.
        x, y = np.argwhere(~np.isnan(thermal[0, :, :]) &
                           ~np.isnan(thermal[1, :, :]) &
                           np.isnan(thermal[10, :, :]))[0]
                           
        numNan = (np.isnan(thermal[:, x, y])).sum()
        maxBand = np.nanargmax(thermal[:, x, y])
        minBand = np.nanargmin(thermal[:, x, y])
        b1MaxB = b1[maxBand, x, y]
        b1MinB = b1[minBand, x, y]
        sortedBandXy = sortedBand[:, x, y]
        amp = abs(b1MaxB - b1MinB)
        self.assertTrue(np.isnan(sortedBandXy[-numNan]))
        self.assertEqual(amp, abs(b1MaxB - b1MinB))
        self.assertEqual(b1Metric.value[x, y], int(abs(b1MaxB - b1MinB)))

    # -------------------------------------------------------------------------
    # testMetricTempMeanWarmest3
    #
    # V2 test successsful
    # -------------------------------------------------------------------------
    def testMetricTempMeanWarmest3(self):

        mm = self.mm
        metric = mm.metricTempMeanWarmest3()[0]
        self.assertTrue(metric.name, 'TempMeanWarmest3')
        thermal, tXref = mm.getBandCube(self.mm._productType.BAND31)
        tSorted = np.sort(thermal, axis=0)
        
        # Start with a location with no NaNs.
        x, y = np.argwhere(~np.isnan(tSorted[11, :, :]))[0]
        
        mean = ((tSorted[9, x, y] + \
                 tSorted[10, x, y] + \
                 tSorted[11, x, y]) / 3.0).astype(int)
        
        self.assertEqual(metric.value[x, y], mean)

        # ---
        # Test the last (greatest) thermal value being NaN.  Ensure that it is
        # not all NaN.
        # ---
        x, y = np.argwhere(~np.isnan(tSorted[0, :, :]) &
                           ~np.isnan(tSorted[1, :, :]) &
                           np.isnan(tSorted[11, :, :]))[0]

        self.assertTrue(np.isnan(tSorted[-1, x, y]))
        
        mean = ((tSorted[8, x, y] + tSorted[9, x, y] + tSorted[10, x, y]) / \
                3.0).astype(int)

        self.assertEqual(metric.value[x, y], mean)

        # Test the last two (greatest) thermal values being NaN.
        x, y = np.argwhere(~np.isnan(tSorted[0, :, :]) &
                           ~np.isnan(tSorted[1, :, :]) &
                           np.isnan(tSorted[10, :, :]))[0]

        self.assertTrue(np.isnan(tSorted[-2, x, y]))

        mean = ((tSorted[7, x, y] + tSorted[8, x, y] + tSorted[9, x, y]) / \
                3.0).astype(int)
                
        self.assertEqual(metric.value[x, y], mean)

        # Test the last three (greatest) thermal values being NaN.
        x, y = np.argwhere(~np.isnan(tSorted[0, :, :]) &
                           ~np.isnan(tSorted[1, :, :]) &
                           np.isnan(tSorted[9, :, :]))[0]

        self.assertTrue(np.isnan(tSorted[-3, x, y]))

        mean = ((tSorted[6, x, y] + tSorted[7, x, y] + tSorted[8, x, y]) / \
                3.0).astype(int)

        self.assertEqual(metric.value[x, y], mean)
        
        # Test all NaN.
        x, y = np.argwhere(np.isnan(tSorted[0, :, :]))[0]
        self.assertTrue(np.isnan(tSorted[0, x, y]))
        self.assertTrue(np.isnan(tSorted[1, x, y]))
        self.assertTrue(np.isnan(tSorted[-1, x, y]))
        self.assertEqual(metric.value[x, y], Band.NO_DATA)
        
    # -------------------------------------------------------------------------
    # testMetricTempMeanGreenest3
    #
    # V2 test successsful
    # -------------------------------------------------------------------------
    def testMetricTempMeanGreenest3(self):

        mm = self.mm
        metric = mm.metricTempMeanGreenest3()[0]
        self.assertTrue(metric.name, 'TempMeanGreenest3')
        ndvi, ndviXref = mm.getNdvi()
        tSorted = np.sort(ndvi, axis=0)
        
        # Start with a location with no NaNs.
        x, y = np.argwhere(~np.isnan(tSorted[11, :, :]))[0]
        
        mean = ((tSorted[9, x, y] + \
                 tSorted[10, x, y] + \
                 tSorted[11, x, y]) / 3.0).astype(int)
        
        self.assertEqual(metric.value[x, y], mean)

        # ---
        # Test the last (greatest) thermal value being NaN.  Ensure that it is
        # not all NaN.
        # ---
        x, y = np.argwhere(~np.isnan(tSorted[0, :, :]) &
                           ~np.isnan(tSorted[1, :, :]) &
                           np.isnan(tSorted[11, :, :]))[0]

        self.assertTrue(np.isnan(tSorted[-1, x, y]))

        mean = ((tSorted[8, x, y] + tSorted[9, x, y] + tSorted[10, x, y]) / \
                3.0).astype(int)

        self.assertEqual(metric.value[x, y], mean)

        # Test the last two (greatest) thermal values being NaN.
        x, y = np.argwhere(~np.isnan(tSorted[0, :, :]) &
                           ~np.isnan(tSorted[1, :, :]) &
                           np.isnan(tSorted[10, :, :]))[0]

        self.assertTrue(np.isnan(tSorted[-2, x, y]))

        mean = ((tSorted[7, x, y] + tSorted[8, x, y] + tSorted[9, x, y]) / \
                3.0).astype(int)

        self.assertEqual(metric.value[x, y], mean)

        # Test the last three (greatest) thermal values being NaN.
        x, y = np.argwhere(~np.isnan(tSorted[0, :, :]) &
                           ~np.isnan(tSorted[1, :, :]) &
                           np.isnan(tSorted[9, :, :]))[0]

        self.assertTrue(np.isnan(tSorted[-3, x, y]))

        mean = ((tSorted[6, x, y] + tSorted[7, x, y] + tSorted[8, x, y]) / \
                3.0).astype(int)

        self.assertEqual(metric.value[x, y], mean)

        # Test all NaN.
        x, y = np.argwhere(np.isnan(tSorted[0, :, :]))[0]
        self.assertTrue(np.isnan(tSorted[0, x, y]))
        self.assertTrue(np.isnan(tSorted[1, x, y]))
        self.assertTrue(np.isnan(tSorted[-1, x, y]))
        self.assertEqual(metric.value[x, y], Band.NO_DATA)

    # -------------------------------------------------------------------------
    # testMod09
    #
    # V2 test successsful
    # -------------------------------------------------------------------------
    def testMod09(self):
        
        inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD09A1')

        outDir = Path('/explore/nobackup/people/rlgill' +      
                      '/SystemTesting/modis-vcf/MOD09A')

        pt = ProductTypeMod09A(inDir,
                               MetricsTestCase._realInDir)
        
        mm = Metrics(MetricsTestCase._validTileId,
                     MetricsTestCase._validYear,
                     pt,
                     outDir,
                     MetricsTestCase._logger)
        
        # Test metricUnsortedMonthlyBands.
        metrics = mm.metricUnsortedMonthlyBands()
        self.assertEqual(len(metrics), 96)
        self.assertEqual(metrics[0].value.shape, (4800, 4800))

        self.assertEqual(metrics[0].name,
                         'UnsortedMonthlyBands-Band_1-Day-2019065')
                         
        # Test metricBandReflMedian with NaNs.
        metrics = mm.metricBandReflMedian()
        self.assertEqual(len(metrics), 8)
        metric = metrics[0]
        self.assertEqual(metric.name, 'BandReflMedian-Band_1')
        
        b1, xref = mm.getBandCube(pt.BAND1)
        numNotNan = (~np.isnan(b1)).sum(axis=0)
        x, y = np.argwhere(numNotNan == 10)[0]
        self.assertEqual(metric.value[x, y], 1031)
        
    # -------------------------------------------------------------------------
    # testMetricAmpBandRefl
    #
    # V2 test successsful
    # -------------------------------------------------------------------------
    def testMetricAmpBandRefl(self):
        
        inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD09A1')

        outDir = Path('/explore/nobackup/people/rlgill' +      
                      '/SystemTesting/modis-vcf/MOD09A')

        pt = ProductTypeMod09A(inDir,
                               MetricsTestCase._realInDir)
        
        mm = Metrics(MetricsTestCase._validTileId,
                     MetricsTestCase._validYear,
                     pt,
                     outDir,
                     MetricsTestCase._logger)
        
        metrics = mm.metricAmpBandRefl()
        self.assertEqual(len(metrics), 8)
        self.assertEqual(metrics[0].value.shape, (4800, 4800))
        self.assertEqual(metrics[0].name, 'AmpBandRefl-Band_1')
        b1, xref = mm.getBandCube(pt.BAND1)

        # Test no NaNs.
        x, y = np.argwhere(~np.isnan(b1[11, :, :]))[0]
        maxBand = np.nanargmax(b1[:, x, y])
        minBand = np.nanargmin(b1[:, x, y])
        self.assertEqual(maxBand, 4)
        self.assertEqual(minBand, 0)

        b1MaxB = b1[maxBand, x, y]
        b1MinB = b1[minBand, x, y]
        self.assertEqual(b1MaxB, 1506.5)
        self.assertAlmostEqual(b1MinB, 1033.333333333)
        amp = int(b1MaxB - b1MinB)
        self.assertEqual(metrics[0].value[x, y], amp)
        
        # Test least two valid values.  Sorting makes identification easier.
        bSorted = np.sort(b1, axis=0)

        x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
                           ~np.isnan(bSorted[1, :, :]) &
                           np.isnan(bSorted[11, :, :]))[0]

        self.assertTrue(np.isnan(bSorted[-1, x, y]))
        maxBand = np.nanargmax(b1[:, x, y])
        minBand = np.nanargmin(b1[:, x, y])
        self.assertEqual(maxBand, 11)
        self.assertEqual(minBand, 5)

        b1MaxB = b1[maxBand, x, y]
        b1MinB = b1[minBand, x, y]
        self.assertEqual(b1MaxB, 1804)
        self.assertEqual(b1MinB, 712)
        amp = int(b1MaxB - b1MinB)
        self.assertEqual(metrics[0].value[x, y], amp)
        
        # ---
        # No case available with only one valid value.
        # Test all NaNs.
        # ---
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertEqual(metrics[0].value[x, y], pt.NO_DATA)
        
    # -------------------------------------------------------------------------
    # testMod09SplitDays
    # -------------------------------------------------------------------------
    # def testMod09SplitDays(self):
    #
    #     inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD09A1')
    #
    #     outDir = Path('/explore/nobackup/people/rlgill' +
    #                   '/SystemTesting/modis-vcf/MOD09A')
    #
    #     pt = ProductTypeMod09A(inDir,
    #                            MetricsTestCase._realInDir)
    #
    #     mm = Metrics(MetricsTestCase._validTileId,
    #                  MetricsTestCase._validYear,
    #                  pt,
    #                  outDir,
    #                  MetricsTestCase._logger)
    #
    #     numSplits, allDays = mm._getNumSplits()
    #     splitDays = mm._splitDays(numSplits, allDays)
    #
    #     expSplits = [['2019065', '2019073', '2019081', '2019089'],
    #                  ['2019097', '2019105', '2019113', '2019121'],
    #                  ['2019129', '2019137', '2019145', '2019153'],
    #                  ['2019161', '2019169', '2019177', '2019185'],
    #                  ['2019193', '2019201', '2019209', '2019217'],
    #                  ['2019225', '2019233', '2019241', '2019249'],
    #                  ['2019257', '2019265', '2019273', '2019281'],
    #                  ['2019289', '2019297', '2019305', '2019313'],
    #                  ['2019321', '2019329', '2019337', '2019345'],
    #                  ['2019353', '2019361', '2020001', '2020009'],
    #                  ['2020017', '2020025', '2020033', '2020041'],
    #                  ['2020049', '2020057']]
    #
    #     self.assertEqual(splitDays, expSplits)
        
    # -------------------------------------------------------------------------
    # testMod44SplitDays
    # -------------------------------------------------------------------------
    # def testMod44SplitDays(self):
    #
    #     mm = self.mm
    #
    #     numSplits, allDays = mm._getNumSplits()
    #     splitDays = mm._splitDays(numSplits, allDays)
    #
    #     expSplits = [['2019065', '2019081'], ['2019097', '2019113'],
    #                  ['2019129', '2019145'], ['2019161', '2019177'],
    #                  ['2019193', '2019209'], ['2019225', '2019241'],
    #                  ['2019257', '2019273'], ['2019289', '2019305'],
    #                  ['2019321', '2019337'], ['2019353', '2020001'],
    #                  ['2020017', '2020033'], ['2020049']]
    #
    #     self.assertEqual(splitDays, expSplits)
        
    # -------------------------------------------------------------------------
    # testMod09B31
    #
    # V2 test successsful
    # -------------------------------------------------------------------------
    def testMod09B31(self):
        
        inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD09A1')

        outDir = Path('/explore/nobackup/people/rlgill' +      
                      '/SystemTesting/modis-vcf/MOD09A')

        pt = ProductTypeMod09A(inDir,
                               MetricsTestCase._realInDir)
        
        mm = Metrics(MetricsTestCase._validTileId,
                     MetricsTestCase._validYear,
                     pt,
                     outDir,
                     MetricsTestCase._logger)
        
        b31, xref = mm.getBandCube(pt.BAND31)
        
    # -------------------------------------------------------------------------
    # testMetricBandReflMedianTemp
    # -------------------------------------------------------------------------
    def testMetricBandReflMedianTemp(self):
        
        mm = self.mm
        metrics = mm.metricBandReflMedianTemp()
        self.assertEqual(len(metrics), 8)
        self.assertEqual(metrics[0].value.shape, (4800, 4800))
        self.assertEqual(metrics[0].name, 'BandReflMedianTemp-Band_1')

        thermal, tXref = mm.getBandCube(self._productType.BAND31)
        tSorted = np.sort(thermal, axis=0)
        b1, xref = mm.getBandCube(self.mm._productType.BAND1)
        bSorted = np.sort(b1, axis=0)
        bSortedByT = self.mm._sortByThermal(b1)
        
        # Case 1:  band with no NaNs and thermal with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]))[0]
        index = int(12 / 2)
        exp = int(bSortedByT[index, x, y])
        self.assertEqual(metrics[0].value[x, y], exp)
        
        # ---
        # Case 2:  band with no NaNs and thermal with some NaNs
        # Unable to find this case.
        # x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
        #                    np.isnan(tSorted[11, :, :]))
        # ---
                           
        # ---
        # Case 3:  band with some NaNs and thermal with no NaNs
        # Unable to find this case.
        # x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
        #                    ~np.isnan(bSorted[1, :, :]) &
        #                    np.isnan(bSorted[11, :, :]) &
        #                    ~np.isnan(tSorted[11, :, :]))[0]
        # ---
                           
        # Case 4:  band with some NaNs and thermal with some NaNs
        x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
                           ~np.isnan(bSorted[1, :, :]) &
                           np.isnan(bSorted[11, :, :]) &
                           np.isnan(tSorted[11, :, :]))[0]
        
        numThermNotNan = (~np.isnan(thermal[:, x, y])).sum(axis=0)
        index = int(numThermNotNan / 2)  # 5
        exp = int(bSortedByT[index, x, y])
        self.assertEqual(metrics[0].value[x, y], exp)
        
        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertEqual(metrics[0].value[x, y], mm._productType.NO_DATA)
        
        # Case 6:  thermal all NaN
        x, y = np.argwhere(np.isnan(tSorted[0, :, :]))[0]
        self.assertEqual(metrics[0].value[x, y], mm._productType.NO_DATA)
        