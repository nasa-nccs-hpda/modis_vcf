
import logging
import os
from pathlib import Path
import sys
import shutil
import tempfile
import unittest

import pytest

import numpy as np

from osgeo import gdal
from osgeo import gdal_array
from osgeo.osr import SpatialReference

from modis_vcf.model import Settings
from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.CompositeDayFile import CompositeDayFile
from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44


# -----------------------------------------------------------------------------
# class MetricsTestCase
#
# pytest tests/test_Metrics.py
# pytest tests/test_Metrics.py::MetricsTestCase::testDirs
#
# TODO: update all tests for new set up
# TODO: complete metrics 
# TODO: test NDVI-based metrics with NDVI band
# -----------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.skipif(Settings.MOD44_COLLECTION != '61', reason='reads Collection 6.1 .hdf inputs: set VCF_MOD44_COLLECTION=61')
class MetricsTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUp
    # -------------------------------------------------------------------------
    def setUp(self):

        # Logger
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)

        if (not self._logger.hasHandlers()):

            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            self._logger.addHandler(ch)

        # MOD44
        self.h09v05 = 'h09v05'
        self.year2019 = 2019
        
        self._mod44InDir = \
            Path('/css/modis/Collection6.1/L3/MOD44B-VCF/dev')

        self.productTypeMod44 = ProductTypeMod44(self._mod44InDir)

        self._mod44OutDir = Path(tempfile.mkdtemp(prefix='vcf_test_'))  # removed after the test
        self.addCleanup(shutil.rmtree, str(self._mod44OutDir), True)
        self._mod44OutDir.mkdir(exist_ok=True)

        self.mmMod44 = Metrics(self.h09v05,
                               self.year2019,
                               self.productTypeMod44,
                               self._mod44OutDir,
                               self._logger)
                           
        self.days = [(2019,  65), (2019,  97), (2019, 129), (2019, 161),
                    (2019, 193), (2019, 225), (2019, 257), (2019, 289),
                    (2019, 321), (2019, 353), (2020,  17), (2020, 49)]

    # -------------------------------------------------------------------------
    # findStacksForApplyThreshold
    # -------------------------------------------------------------------------
    def findStacksForApplyThreshold(self, band: np.ndarray, threshold: int):
        
        # ---
        # Find stacks to test.
        # Default threshold is 3.  Loop through pixel by pixel to identify
        # one stack with more than 3 NaNs, one stack with 3 NaNs, and one
        # stack with fewer than 3 NaNs.
        # ---
        ltThresh = None
        eqThresh = None
        gtThresh = None

        for c in range(ProductTypeMod44.COLS):
            
            for r in range(ProductTypeMod44.ROWS):

                stack = band[:, c, r]
                numNan = np.count_nonzero(np.isnan(stack))
                
                if numNan == threshold:
                    
                    if not eqThresh:
                        eqThresh = (c, r)
                    
                elif numNan > threshold and numNan < 12:
                    
                    if not gtThresh:
                        gtThresh = (c, r)
                    
                elif numNan < threshold:
                    
                    ltThresh = (c, r)
                    
            if ltThresh and eqThresh and gtThresh:
                break

        print('LT stack:', ltThresh)
        print('EQ stack:', eqThresh)
        print('GT stack:', gtThresh)

    # -------------------------------------------------------------------------
    # getRandomCoords
    # -------------------------------------------------------------------------
    def getRandomCoords(self) -> list:
        
        rng = np.random.default_rng()
        rows = rng.integers(low=0, high=ProductTypeMod44.ROWS, size=5)
        cols = rng.integers(low=0, high=ProductTypeMod44.COLS, size=5)
        coords = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)
        return coords
        
    # -------------------------------------------------------------------------
    # toTif
    # -------------------------------------------------------------------------
    def toTif(self, raster: np.ndarray, outName: Path) -> None:
        
        dataType = \
            gdal_array.NumericTypeCodeToGDALTypeCode(raster.dtype)
    
        ds = gdal.GetDriverByName('GTiff').Create(
            str(outName),
            raster.shape[1],
            raster.shape[2],
            raster.shape[0],
            dataType,
            options=['COMPRESS=LZW', 'BIGTIFF=YES'])

        outIndex = 0
        
        for i in range(raster.shape[0]):
            
            outIndex += 1
            gdBand = ds.GetRasterBand(outIndex)
            gdBand.WriteArray(raster[i, :, :])
            gdBand.FlushCache()
            gdBand = None

        ds = None
        
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
                                       
        # ---
        # Apply threshold is obsolete because there are no more NaNs.
        # Get a band without applying the threshold
        # ---
        # bandName = ProductTypeMod44.BAND1
        # mm = self.mmMod44
        # band, bXref = mm.getBandCube(bandName)
        # threshold = 3
        #
        # # ---
        # # Use MetricsTestCase.findStacksForApplyThreshold() to discover these.
        # # self.findStacksForApplyThreshold(band, threshold)
        # # ---
        # ltx = 0
        # lty = 4799
        # eqx = 0
        # eqy = 232
        # gtx = 0
        # gty = 292
        #
        # # These are how many NaNs are expected at these locations.
        # self.assertEqual(np.count_nonzero(np.isnan(band[:, ltx, lty])), 0)
        #
        # testDays = []
        #
        # for day in self.days:
        #
        #     name = 'MOD44-' + mm._tid + '-' + str(day[0]) + \
        #            str(day[1]).zfill(3) + '-' + bandName + '.bin'
        #
        #     fName = mm._dayDir / name
        #
        #     raster = np.fromfile(fName, dtype=np.int16). \
        #              reshape(ProductTypeMod44.ROWS, ProductTypeMod44.COLS)
        #
        #     testDays.append(raster)
        #
        # self.assertEqual(np.count_nonzero(np.isnan(band[:, eqx, eqy])), 3)
        # self.assertEqual(np.count_nonzero(np.isnan(band[:, gtx, gty])), 4)
        #
        # # ---
        # # Now apply the threshold.
        # # This is a kludgey way to change the threshold, rather than
        # # implementing an accessor.
        # # ---
        # threshMm = mm
        # threshMm._nanThreshold = threshold
        # bandThresh, bXref = threshMm.getBandCube(bandName)
        #
        # self.assertFalse(np.isnan(np.nanmax(bandThresh)))
        #
        # self.assertEqual(np.count_nonzero( \
        #     np.isnan(bandThresh[:, ltx, lty])), 0)
        #
        # self.assertEqual(np.count_nonzero( \
        #     np.isnan(bandThresh[:, eqx, eqy])), 3)
        #
        # NUM_MOD44_SPLITS = 12
        #
        # self.assertEqual(np.count_nonzero(np.isnan(bandThresh[:, gtx, gty])),
        #                  NUM_MOD44_SPLITS)
        #
        # self.assertTrue(np.array_equal(band[:, ltx, lty],
        #                                bandThresh[:, ltx, lty],
        #                                equal_nan=True))
        #
        # self.assertTrue(np.array_equal(band[:, eqx, eqy],
        #                                bandThresh[:, eqx, eqy],
        #                                equal_nan=True))
        #
        # self.assertFalse(np.array_equal(band[:, gtx, gty],
        #                                 bandThresh[:, gtx, gty],
        #                                 equal_nan=True))

    # -------------------------------------------------------------------------
    # testDirs
    # -------------------------------------------------------------------------
    def testDirs(self):

        self.assertEqual(self.mmMod44._outDir, self.mmMod44._outDir)
        
        self.assertEqual(self.mmMod44._dayDir, 
                         self.mmMod44._outDir / Path('1-Days'))
                         
        self.assertEqual(self.mmMod44._compDir, 
                         self.mmMod44._outDir / Path('2-Composites'))
        
    # -------------------------------------------------------------------------
    # testGetBandCube
    # -------------------------------------------------------------------------
    def testGetBandCube(self):

        b5, b5Xref = self.mmMod44.getBandCube(ProductTypeMod44.BAND5)
        self.assertEqual(b5.shape, (8, 4800, 4800))
        self.assertEqual(b5.dtype, np.float64)

        days = ['2019065', '2019113', '2019161', '2019209', '2019257',
                '2019305', '2019353', '2020033']

        self.assertEqual(list(b5Xref.keys()), days)
        
        # Ensure bands no longer have NaNs or no-data values.
        mm = Metrics('h12v02',
                     self.year2019,
                     self.productTypeMod44,
                     self._mod44OutDir,
                     self._logger)

        b6, b6Xref = mm.getBandCube(ProductTypeMod44.BAND6)
        self.assertFalse(np.isnan(b6).any())
        self.assertFalse((b6 == -10001).any())
        
    # -------------------------------------------------------------------------
    # testGetMetricBand
    # -------------------------------------------------------------------------
    def testGetMetricBand(self):
        
        METRIC_TITLE = 'BandReflMin'
        METRIC_NAME = 'metric' + METRIC_TITLE
        mm = self.mmMod44
        b1: np.ndarray = mm.getMetricBand(METRIC_NAME, ProductTypeMod44.BAND1)
        self.assertEqual(b1.shape, (4800, 4800))
        
        METRIC_TITLE = 'UnsortedMonthlyBands'
        METRIC_NAME = 'metric' + METRIC_TITLE
        mm = self.mmMod44

        b2: np.ndarray = mm.getMetricBand(METRIC_NAME, 
                                          ProductTypeMod44.BAND2,
                                          day='2019161')
                                          
        self.assertEqual(b2.shape, (4800, 4800))
        
        b3: np.ndarray = mm.getMetricBand(METRIC_NAME, 
                                          ProductTypeMod44.NDVI,
                                          day='2019161')
                                          
        self.assertEqual(b3.shape, (4800, 4800))
        
    # -------------------------------------------------------------------------
    # testGetMetricFromRf
    # -------------------------------------------------------------------------
    def testGetMetricFromRf(self):
        
        mm = self.mmMod44
        b = mm.getMetricFromRf('UnsortedMonthlyBands-Band_6-Day_2019257')
        self.assertEqual(b.shape, (4800, 4800))
        b = mm.getMetricFromRf('UnsortedMonthlyBands-NDVI-Day_2020033')
        self.assertEqual(b.shape, (4800, 4800))
        
    # -------------------------------------------------------------------------
    # testCompositeValues
    # -------------------------------------------------------------------------
    def testCompositeValues(self):

        mm = Metrics('h12v02',
                     self.year2019,
                     self.productTypeMod44,
                     self._mod44OutDir,
                     self._logger)

        b5Name = ProductTypeMod44.BAND5
        b5, b5Xref = mm.getBandCube(b5Name)
        self.assertEqual(np.count_nonzero(np.isnan(b5[:, 0, 0])), 0)

        daysSought = [(2019,  65), (2019, 113), (2019, 161), (2019, 209),
                      (2019, 257), (2019, 305), (2019, 353), (2020,  33)]

        testDays = []
        
        for year, day in daysSought:
            
            name = 'MOD44-' + mm._tid + '-' + str(year) + \
                   str(day).zfill(3) + '-' + b5Name + '.bin'
                   
            fName = mm._compDir / name
            
            raster = np.fromfile(fName, dtype=np.int16). \
                     reshape(ProductTypeMod44.ROWS, ProductTypeMod44.COLS)
            
            testDays.append(raster)

        for i in range(8):
            self.assertEqual(b5[i, 0, 0], testDays[i][0, 0])

    # -------------------------------------------------------------------------
    # testGetDayXref
    # -------------------------------------------------------------------------
    def testGetDayXref(self):

        # Get a combined band to ensure the xref was written.
        b5, xref = self.mmMod44.getBandCube(ProductTypeMod44.BAND5)
        print('xref: ' + str(xref))
        self.assertTrue('2019065' in xref)
        self.assertTrue('2020033' in xref)
        self.assertEqual(xref['2019209'], 3)
        
    # -------------------------------------------------------------------------
    # testRegistration
    # -------------------------------------------------------------------------
    def testRegistration(self):
        
        print('Available metrics:', self.mmMod44.availableMetrics)
        self.mmMod44.printAvailableMetrics()
        
    # -------------------------------------------------------------------------
    # testGetYearForDay
    # -------------------------------------------------------------------------
    def testGetYearForDay(self):
        
        year = self.mmMod44.getYearForDay(65)
        self.assertEqual(year, self.mmMod44._year)

        year = self.mmMod44.getYearForDay(257)
        self.assertEqual(year, self.mmMod44._year)

        year = self.mmMod44.getYearForDay(33)
        self.assertEqual(year, self.mmMod44._year + 1)

    # -------------------------------------------------------------------------
    # testSortByNdvi
    # -------------------------------------------------------------------------
    def testSortByNdvi(self):

        mm = self.mmMod44
        band, bXref = mm.getBandCube(ProductTypeMod44.BAND3)
        bSorted: np.ndarray = np.sort(band, axis=0)
        ndvi, ndviXref = mm.getNdvi()
        nSorted = np.sort(ndvi, axis=0)

        # Test NDVI sort. 
        ndviSortedBand: np.ndarray = mm._sortByNDVI(band)
        self.assertFalse((ndviSortedBand == band).all())
        
        x = 0
        y = 0
        maxIndex = np.argmax(ndvi[:, x, y])
        self.assertEqual(maxIndex, 1)
        self.assertAlmostEqual(ndvi[maxIndex, x, y], 298.7625221)
        self.assertEqual(ndviSortedBand[-1, x, y], band[maxIndex, x, y])
        
        x = 2100
        y = 1200
        maxIndex = np.argmax(ndvi[:, x, y])
        self.assertEqual(maxIndex, 2)
        self.assertAlmostEqual(ndvi[maxIndex, x, y], 151.12540193)
        self.assertEqual(ndviSortedBand[-1, x, y], band[maxIndex, x, y])
        
        # Test h12v02
        mm = Metrics('h12v02',
                     self.year2019,
                     self.productTypeMod44,
                     self._mod44OutDir,
                     self._logger)

        band, bXref = mm.getBandCube(ProductTypeMod44.BAND3)
        bSorted: np.ndarray = np.sort(band, axis=0)
        ndvi, ndviXref = mm.getNdvi()
        nSorted = np.sort(ndvi, axis=0)

        ndviSortedBand: np.ndarray = mm._sortByNDVI(band)
        self.assertFalse((ndviSortedBand == band).all())
        
        x = 0
        y = 0
        maxIndex = np.argmax(ndvi[:, x, y])
        self.assertEqual(maxIndex, 1)
        self.assertAlmostEqual(ndvi[maxIndex, x, y], 0)
        self.assertEqual(ndviSortedBand[-1, x, y], band[maxIndex, x, y])
        
        x = 2100
        y = 1200
        maxIndex = np.argmax(ndvi[:, x, y])
        self.assertEqual(maxIndex, 2)
        self.assertAlmostEqual(ndvi[maxIndex, x, y], 562.89707751)
        self.assertEqual(ndviSortedBand[-1, x, y], band[maxIndex, x, y])

    # -------------------------------------------------------------------------
    # testSortByThermal
    # -------------------------------------------------------------------------
    def testSortByThermal(self):

        mm = self.mmMod44
        band, bXref = mm.getBandCube(ProductTypeMod44.BAND3)
        bSorted: np.ndarray = np.sort(band, axis=0)

        tSortedBand: np.ndarray = mm._sortByThermal(band)
        self.assertFalse((tSortedBand == band).all())

        thermal, tXref = mm.getBandCube(ProductTypeMod44.BAND31)
        tSorted = np.sort(thermal, axis=0)

        x = 0
        y = 0
        maxIndex = np.argmax(thermal[:, x, y])
        self.assertEqual(maxIndex, 3)
        self.assertAlmostEqual(thermal[maxIndex, x, y], 32349)
        self.assertEqual(tSortedBand[-1, x, y], band[maxIndex, x, y])

        x = 2100
        y = 1200
        maxIndex = np.argmax(thermal[:, x, y])
        self.assertEqual(maxIndex, 2)
        # Period 3 (DOY 161, 177, 193): DOY 193 fails QA (cloud adjacency), so the composite is the mean of DOY 161 and 177,
        # (31612 + 31806) / 2 = 31709 (raw inputs read 2026-11-11; the old value 27941 predates the current QA mask).
        self.assertAlmostEqual(thermal[maxIndex, x, y], 31709)
        self.assertEqual(tSortedBand[-1, x, y], band[maxIndex, x, y])

    # -------------------------------------------------------------------------
    # The tests of the metrics no product model uses (BandReflMax / Median / Min and their Greenness and Temp forms,
    # Lowest / Warmest N mean, Amp*, TempMean*) were retired in 2026 (v1.9.0): they used an earlier getMetric API.
    # The product predictors are tested in test_CMirrorMetrics.py and test_Composites_Metrics.py.
    # -------------------------------------------------------------------------
    # qaBits
    # -------------------------------------------------------------------------
    def qaBits(self):
        
        pt = self.productTypeMod44
        outDir = self._mod44OutDir
        tid = 'h12v02'
        self._logger.setLevel(logging.WARNING)
        
        mm = Metrics(tid,
                     self.year2019,
                     pt,
                     outDir,
                     self._logger)
                     
        # ---
        # Mark wants information for the day associated with the minimum
        # NDVI value for certain points.
        # ---
        ndvi, nXref = mm.getNdvi()
        ascIndexes = np.argsort(ndvi, axis=0)

        p1 = (3965, 1010)
        p2 = (3965, 1009)
        d1Index = ascIndexes[0, p1[0], p1[1]]
        d2Index = ascIndexes[0, p2[0], p2[1]]
        yd1 = list(nXref.keys())[list(nXref.values()).index(d1Index)]
        yd2 = list(nXref.keys())[list(nXref.values()).index(d2Index)]
        yearDays = [yd1, yd2]
                     
        for bandName in pt.BANDS:
    
            for yearDay in yearDays:
            
                year = int(yearDay[:4])
                day = int(yearDay[5:])
                
                cdf = CompositeDayFile().initFromParams(mm._productType, 
                                                        bandName,
                                                        mm._tid, 
                                                        year, 
                                                        day, 
                                                        mm._compDir,
                                                        logger=mm._logger,
                                                        dayDir=mm._dayDir)
        
                for cdfYear, cdfDay in cdf._getDaysToFind():

                    bdf = BandDayFile().initFromParams(cdf.productType,
                                                       cdf.bandName,
                                                       cdf.tid,
                                                       cdfYear,
                                                       cdfDay,
                                                       cdf._outDir,
                                                       cdf._logger)
    
                    state, dtype = \
                        bdf._readSubdataset(pt.STATE, 
                                            False,
                                            productType=bdf.productType)
                                            
                    solz, dType = \
                        bdf._readSubdataset(pt.SOLZ,
                                            productType=bdf.productType)
        
                    solz = (solz * pt.solarZenithScaleFactor).astype(np.int16)

                    if yearDay == yd1:
                        
                        v = state[p1[0], p1[1]]
                        sz = solz[p1[0], p1[1]]
                        print(p1, bandName, cdfYear, cdfDay, bin(v), sz)
                        
                    else:

                        v = state[p2[0], p2[1]]
                        sz = solz[p2[0], p2[1]]
                        print(p2, bandName, cdfYear, cdfDay, bin(v), sz)
