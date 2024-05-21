
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


# -----------------------------------------------------------------------------
# class MetricsTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_Metrics
# python -m unittest modis_vcf.model.tests.test_Metrics.MetricsTestCase.testGetSortKeys
#
# TODO:  metricAmpGreenestBandRefl, metricAmpWarmestBandRefl,
#        metricTempMeanWarmest3
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
    def testSort(self):

        mm = self.mm
        b1: np.ndarray = mm.getBandCube(Pair.BAND1)
        sortedBand: np.ndarray = np.sort(b1, axis=0)
        self.assertFalse((sortedBand == b1).all())

        # Test NDVI sort.
        ndviSortedBand: np.ndarray = mm._sortByNDVI(b1)
        self.assertFalse((ndviSortedBand == b1).all())

        # Find the largest NDVI value at the location.
        x = 21
        y = 12
        maxIndex = mm.getNdvi()[:, x, y].argmax()
        self.assertEqual(maxIndex, 1)
        self.assertAlmostEqual(mm.getNdvi()[maxIndex, x, y], 364.26116838)
        self.assertEqual(ndviSortedBand[-1, x, y], b1[maxIndex, x, y])

        # Test thermal sort.
        thermSortedBand: np.ndarray = mm._sortByThermal(b1)
        self.assertFalse((thermSortedBand == b1).all())

        # Find the largest thermal value at the location.
        x = 2100
        y = 1200
        raw31 = mm._cbbd.getBand(Pair.BAND31)
        combined31: np.ndarray = mm._combine(raw31)[0]
        maxIndex = combined31[:, x, y].argmax()
        self.assertEqual(thermSortedBand[-1, x, y], b1[maxIndex, x, y])
        
        # A NaN case was found.
        x = 100
        y = 1001

    # -------------------------------------------------------------------------
    # testMetricAmpGreenestBandRefl
    # -------------------------------------------------------------------------
    def testMetricAmpGreenestBandRefl(self):

        x = 100
        y = 1001
        mm = self.mm
        b1: np.ndarray = mm.getBandCube(Pair.BAND1)
        ndvi = mm.getNdvi()
        
        maxBand = np.nanargmax(ndvi[:, x, y])
        minBand = np.nanargmin(ndvi[:, x, y])
        self.assertEqual(maxBand, 2)
        self.assertEqual(minBand, 0)
        
        b1MaxB = b1[maxBand, x, y]
        b1MinB = b1[minBand, x, y]
        self.assertEqual(b1MaxB, 1550.0)
        self.assertEqual(b1MinB, 1256.0)
        
        ndviSortedBand: np.ndarray = mm._sortByNDVI(b1)
        sortedBandXy = ndviSortedBand[:, x, y]
        amp = b1MaxB - b1MinB
        self.assertTrue(np.isnan(sortedBandXy[-1]))
        self.assertEqual(b1MaxB, sortedBandXy[-2])
        self.assertEqual(b1MinB, sortedBandXy[0])
        self.assertEqual(amp, b1MaxB - b1MinB)
        
        metric = mm.metricAmpGreenestBandRefl()
        b1Metric = metric[0]
        self.assertTrue(b1Metric.name, 'AmpGreenestBandRefl-Band_1')
        self.assertEqual(b1Metric.value[x, y], b1MaxB - b1MinB)

    # -------------------------------------------------------------------------
    # testMetricAmpWarmestBandRefl
    # -------------------------------------------------------------------------
    def testMetricAmpWarmestBandRefl(self):

        x = 100
        y = 1001
        mm = self.mm
        b1: np.ndarray = mm.getBandCube(Pair.BAND1)
        thermal = mm.getBandCube(Pair.BAND31)

        maxBand = np.nanargmax(thermal[:, x, y])
        minBand = np.nanargmin(thermal[:, x, y])
        self.assertEqual(maxBand, 4)
        self.assertEqual(minBand, 8)

        b1MaxB = b1[maxBand, x, y]
        b1MinB = b1[minBand, x, y]
        self.assertEqual(b1MaxB, 1575.0)
        self.assertEqual(b1MinB, 1412.0)

        sortedBand: np.ndarray = mm._sortByThermal(b1)
        sortedBandXy = sortedBand[:, x, y]
        amp = b1MaxB - b1MinB
        self.assertTrue(np.isnan(sortedBandXy[-1]))
        self.assertEqual(b1MaxB, sortedBandXy[-2])
        self.assertEqual(b1MinB, sortedBandXy[0])
        self.assertEqual(amp, b1MaxB - b1MinB)

        metric = mm.metricAmpWarmestBandRefl()
        b1Metric = metric[0]
        self.assertTrue(b1Metric.name, 'AmpWarmestBandRefl-Band_1')
        self.assertEqual(b1Metric.value[x, y], b1MaxB - b1MinB)

    # -------------------------------------------------------------------------
    # testMetricTempMeanWarmest3
    # -------------------------------------------------------------------------
    def testMetricTempMeanWarmest3(self):

        mm = self.mm
        metric = mm.metricTempMeanWarmest3()[0]
        self.assertTrue(metric.name, 'TempMeanWarmest3')
        thermal = mm.getBandCube(Pair.BAND31)
        tSorted = np.sort(thermal, axis=0)
        
        # Start with a location with no NaNs.
        x, y = np.argwhere(~np.isnan(tSorted[10, :, :]))[0]
        
        mean = ((tSorted[8, x, y] + \
                 tSorted[9, x, y] + \
                 tSorted[10, x, y]) / 3.0).astype(int)
        
        self.assertEqual(metric.value[x, y], mean)

        # Test the last (greatest) thermal value being NaN.
        x, y = np.argwhere(np.isnan(tSorted[10, :, :]))[0]
        self.assertTrue(np.isnan(tSorted[-1, x, y]))
        mean = ((tSorted[8, x, y] + tSorted[9, x, y]) / 2.0).astype(int)
        self.assertEqual(metric.value[x, y], mean)

        # Test the last two (greatest) thermal values being NaN.
        x, y = np.argwhere(np.isnan(tSorted[9, :, :]))[0]
        self.assertTrue(np.isnan(tSorted[-1, x, y]))
        mean = tSorted[8, x, y].astype(int)
        self.assertEqual(metric.value[x, y], mean)

        # Test the last three (greatest) thermal values being NaN.
        x, y = np.argwhere(np.isnan(tSorted[8, :, :]))[0]
        self.assertTrue(np.isnan(tSorted[-1, x, y]))
        mean = Band.NO_DATA
        self.assertEqual(metric.value[x, y], mean)
        
    # -------------------------------------------------------------------------
    # testMetricTempMeanGreenest3
    #
    # (Pdb) cube[:,0,0]
    # array([29648.5, 30692. , 31089. , 31559.5, 32294.5, 32236.5, 30773. ,
    #        29787. , 28776. , 27932. , 29066.5])
    #
    # (Pdb) self.getNdvi()[:,0,0]
    # array([273.75565611, 332.27445997, 284.29944655, 246.61166765,
    #        205.24691358, 177.7997458 , 179.27608233, 177.40011926,
    #        203.43839542, 212.49127704, 192.83658459])
    #
    # (Pdb) sortedCube[:,0,0]
    # array([29787. , 32236.5, 30773. , 29066.5, 28776. , 32294.5, 27932. ,
    #        31559.5, 29648.5, 31089. , 30692. ])
    # -------------------------------------------------------------------------
    def testMetricTempMeanGreenest3(self):

        mm = self.mm
        metric = mm.metricTempMeanGreenest3()[0]
        self.assertTrue(metric.name, 'TempMeanGreenest3')
        thermal = mm.getBandCube(Pair.BAND31)
        ndvi = mm.getNdvi()
        nSorted = np.sort(ndvi, axis=0)

        # Start with a location with no NaNs.
        x, y = np.argwhere(~np.isnan(nSorted[10, :, :]))[0]
        targetStack = ndvi[:, x, y]  # Isolate the non-nan stack
        maxVal = nSorted[-1, x, y]   # Get the three maximum ndvi values
        maxVal2 = nSorted[-2, x, y]
        maxVal3 = nSorted[-3, x, y]
        maxNPos = np.where(targetStack == maxVal)  # Get pos of max ndvis
        maxNPos2 = np.where(targetStack == maxVal2)
        maxNPos3 = np.where(targetStack == maxVal3)
        maxThermal = thermal[maxNPos, x, y].item()  # Thermal at those pos'ns
        maxThermal2 = thermal[maxNPos2, x, y].item()
        maxThermal3 = thermal[maxNPos3, x, y].item()
        mean = int((maxThermal + maxThermal2 + maxThermal3) / 3.0)
        self.assertEqual(metric.value[x, y], mean)
        
        # Test the last (greatest) NDVI value being NaN.
        x, y = np.argwhere(np.isnan(nSorted[10, :, :]))[0]
        targetStack = ndvi[:, x, y]  # Isolate the non-nan stack
        maxVal = nSorted[9, x, y]   # Get the three maximum ndvi values
        maxVal2 = nSorted[8, x, y]
        maxVal3 = nSorted[7, x, y]
        maxNPos = np.where(targetStack == maxVal)  # Get pos of max ndvis
        maxNPos2 = np.where(targetStack == maxVal2)
        maxNPos3 = np.where(targetStack == maxVal3)
        maxThermal = thermal[maxNPos, x, y].item()  # Thermal at those pos'ns
        maxThermal2 = thermal[maxNPos2, x, y].item()
        maxThermal3 = thermal[maxNPos3, x, y].item()
        mean = int((maxThermal + maxThermal2 + maxThermal3) / 3.0)
        self.assertEqual(metric.value[x, y], mean)
