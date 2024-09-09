
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from osgeo import gdal
from osgeo import gdal_array
from osgeo.osr import SpatialReference

from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.CompositeDayFile import CompositeDayFile
from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.ProductTypeMod09A import ProductTypeMod09A
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44


# -----------------------------------------------------------------------------
# class MetricsTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_Metrics
# python -m unittest modis_vcf.model.tests.test_Metrics.MetricsTestCase.testMod09
#
# TODO: update all tests for new set up
# TODO: complete metrics 
# TODO: test NDVI-based metrics with NDVI band
# -----------------------------------------------------------------------------
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
            Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')

        self._mod44OutDir = Path('/explore/nobackup/people/rlgill' +      
                                 '/SystemTesting/modis-vcf/MOD44')

        self.productTypeMod44 = ProductTypeMod44(self._mod44InDir)

        self.mmMod44 = Metrics(self.h09v05,
                               self.year2019,
                               self.productTypeMod44,
                               self._mod44OutDir,
                               self._logger)
                           
        # MOD09
        self._mod09InDir = \
            Path('/explore/nobackup/projects/ilab/data/MODIS/MOD09A1')

        self._mod09OutDir = Path('/explore/nobackup/people/rlgill' +      
                                 '/SystemTesting/modis-vcf/MOD09A')

        self.productTypeMod09A = \
            ProductTypeMod09A(self._mod09InDir, self._mod44InDir)

        self.mmMod09A = Metrics(self.h09v05,
                                self.year2019,
                                self.productTypeMod09A,
                                self._mod09OutDir,
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
        rows = rng.integers(low=0, high=ProductTypeMod09A.ROWS, size=5)
        cols = rng.integers(low=0, high=ProductTypeMod09A.COLS, size=5)
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

        mm = self.mmMod44
        
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
                                       

        # Get a band without applying the threshold
        bandName = ProductTypeMod44.BAND1
        band, bXref = mm.getBandCube(bandName)
        threshold = 3

        # ---
        # Use MetricsTestCase.findStacksForApplyThreshold() to discover these.
        self.findStacksForApplyThreshold(band, threshold)
        # ---
        ltx = 0
        lty = 4799
        eqx = 0
        eqy = 232
        gtx = 0
        gty = 292

        # These are how many NaNs are expected at these locations.
        self.assertEqual(np.count_nonzero(np.isnan(band[:, ltx, lty])), 0)

        testDays = []
        
        for day in self.days:
            
            name = 'MOD44-' + mm._tid + '-' + str(day[0]) + \
                   str(day[1]).zfill(3) + '-' + bandName + '.bin'
                   
            fName = mm._dayDir / name
            
            raster = np.fromfile(fName, dtype=np.int16). \
                     reshape(ProductTypeMod44.ROWS, ProductTypeMod44.COLS)
            
            testDays.append(raster)

        self.assertEqual(np.count_nonzero(np.isnan(band[:, eqx, eqy])), 3)
        self.assertEqual(np.count_nonzero(np.isnan(band[:, gtx, gty])), 4)

        # Now apply the threshold.
        # This is a kludgey way to change the threshold, rather than
        # implementing an accessor.
        # ---
        threshMm = mm
        threshMm._nanThreshold = threshold
        bandThresh, bXref = threshMm.getBandCube(bandName)
        
        self.assertFalse(np.isnan(np.nanmax(bandThresh)))
        self.assertEqual(np.count_nonzero(np.isnan(bandThresh[:, ltx, lty])), 0)
        self.assertEqual(np.count_nonzero(np.isnan(bandThresh[:, eqx, eqy])), 3)
        
        NUM_MOD44_SPLITS = 12

        self.assertEqual(np.count_nonzero(np.isnan(bandThresh[:, gtx, gty])),
                         NUM_MOD44_SPLITS)

        self.assertTrue(np.array_equal(band[:, ltx, lty], 
                                       bandThresh[:, ltx, lty], 
                                       equal_nan=True))

        self.assertTrue(np.array_equal(band[:, eqx, eqy], 
                                       bandThresh[:, eqx, eqy], 
                                       equal_nan=True))

        self.assertFalse(np.array_equal(band[:, gtx, gty], 
                                        bandThresh[:, gtx, gty], 
                                        equal_nan=True))

    # -------------------------------------------------------------------------
    # testGetBandCube
    # -------------------------------------------------------------------------
    def testGetBandCube(self):

        b5, b5Xref = self.mmMod44.getBandCube(ProductTypeMod44.BAND5)
        self.assertEqual(b5.shape, (12, 4800, 4800))
        self.assertEqual(b5.dtype, np.float64)

        days = ['2019065', '2019097', '2019129', '2019161', '2019193',
                '2019225', '2019257', '2019289', '2019321', '2019353',
                '2020017', '2020049', ]

        self.assertEqual(list(b5Xref.keys()), days)
        
    # -------------------------------------------------------------------------
    # testCompositeValues
    # -------------------------------------------------------------------------
    def testCompositeValues(self):

        mm = self.mmMod44
        b5Name = ProductTypeMod44.BAND5
        b5, b5Xref = mm.getBandCube(b5Name)
        self.assertEqual(np.count_nonzero(np.isnan(b5[:, 0, 0])), 0)

        daysSought = [(2019, 65), (2019, 97), (2019, 129), (2019, 161), 
                      (2019, 193), (2019, 225), (2019, 257), (2019, 289),
                      (2019, 321), (2019, 353), (2020, 17), (2020, 49)]
        
        testDays = []
        
        for year, day in daysSought:
            
            name = 'MOD44-' + mm._tid + '-' + str(year) + \
                   str(day).zfill(3) + '-' + b5Name + '.bin'
                   
            fName = mm._compDir / name
            
            raster = np.fromfile(fName, dtype=np.int16). \
                     reshape(ProductTypeMod44.ROWS, ProductTypeMod44.COLS)
            
            testDays.append(raster)

        for i in range(12):
            self.assertEqual(b5[i, 0, 0], testDays[i][0, 0])

        self.assertEqual(np.count_nonzero(np.isnan(b5[:, 0, 232])), 3)
        self.assertEqual(np.count_nonzero(np.isnan(b5[:, 0, 292])), 4)

    # -------------------------------------------------------------------------
    # testGetDayXref
    # -------------------------------------------------------------------------
    def testGetDayXref(self):

        # Get a combined band to ensure the xref was written.
        b5, xref = self.mmMod44.getBandCube(ProductTypeMod44.BAND5)
        print('xref: ' + str(xref))
        self.assertTrue('2019065' in xref)
        self.assertTrue('2020017' in xref)
        self.assertEqual(xref['2019225'], 5)
        
    # -------------------------------------------------------------------------
    # testRegistration
    # -------------------------------------------------------------------------
    def testRegistration(self):
        
        print('Available metrics:', self.mmMod44.availableMetrics)
        self.mmMod44.printAvailableMetrics()
        
    # -------------------------------------------------------------------------
    # testGetNdvi
    # -------------------------------------------------------------------------
    def testGetNdvi(self):
        
        # ---
        # This was a problem case.
        # ---
        mm = Metrics('h12v02', 
                     self.year2019, 
                     self.productTypeMod09A, 
                     self._mod09OutDir, 
                     self._logger)
        
        # Remove existing files.
        ndviName = mm._metricsDir / 'MOD09A-NDVI.bin'
        ndviName.unlink(missing_ok=True)
        
        ndvi, nXref = mm.getNdvi()
        b1, xref = mm.getBandCube(self.productTypeMod09A.BAND1)
        b2, xref = mm.getBandCube(self.productTypeMod09A.BAND2)
        
        self.assertTrue(np.isnan(b1).any())
        self.assertTrue(np.isnan(b2).any())
        self.assertTrue(np.isnan(ndvi).any())

        # ---
        # Test MOD09A.  Elsewhere, NDVI had no NaNs.
        # Remove existing file.
        # ---
        mm = self.mmMod09A
        name = mm._metricsDir / (mm._productType._productType + '-NDVI.bin')
        name.unlink(missing_ok=True)

        ndvi, ndviXref = mm.getNdvi()
        b1, b1Xref = mm.getBandCube(self.productTypeMod09A.BAND1)
        b2, b2Xref = mm.getBandCube(self.productTypeMod09A.BAND2)
        
        self.assertTrue(np.isnan(b1).any())
        self.assertTrue(np.isnan(b2).any())
        self.assertTrue(np.isnan(ndvi).any())
                
        # Remove existing file.
        mm = self.mmMod44
        name = mm._metricsDir / (mm._productType._productType + '-NDVI.bin')
        name.unlink(missing_ok=True)

        ndvi, ndviXref = mm.getNdvi()
        b1, b1Xref = mm.getBandCube(self.productTypeMod44.BAND1)
        b2, b2Xref = mm.getBandCube(self.productTypeMod44.BAND2)
        
        self.assertTrue(np.isnan(b1).any())
        self.assertTrue(np.isnan(b2).any())
        self.assertTrue(np.isnan(ndvi).any())

        # Ensure division by zero works as expected.
        d, r, c = np.argwhere(b1 + b2 == 0)[0]
        self.assertEqual(ndvi[d, r, c], 0)
        
        # Ensure NaNs in bands one or two works as expected.
        d, r, c = np.argwhere(np.isnan(b1))[0]
        self.assertTrue(np.isnan(ndvi[d, r, c]))

        d, r, c = np.argwhere(np.isnan(b2))[0]
        self.assertTrue(np.isnan(ndvi[d, r, c]))
        
        # Test some values.
        d, r, c = (0, 0, 0)
        v1 = b1[d, r, c]
        v2 = b2[d, r, c]
        exp = ((v2 - v1) / (v2 + v1)) * 1000
        self.assertEqual(ndvi[d, r, c], exp)
        
        d, r, c = (2, 1, 12)
        v1 = b1[d, r, c]
        v2 = b2[d, r, c]
        exp = ((v2 - v1) / (v2 + v1)) * 1000
        self.assertEqual(ndvi[d, r, c], exp)

        d, r, c = (2, 11, 2)
        v1 = b1[d, r, c]
        v2 = b2[d, r, c]
        exp = ((v2 - v1) / (v2 + v1)) * 1000
        self.assertEqual(ndvi[d, r, c], exp)

        # ---
        # Test threshold.
        #
        # From running findStacksForApplyThreshold(), we know that
        # LT stack: (0, 0)
        # EQ stack: (0, 232)
        # GT stack: None
        # self.findStacksForApplyThreshold(ndviNoThr)
        # ---
        ndviNoThr, ndviXref = mm.getNdvi()
        self.assertEqual(np.count_nonzero(np.isnan(ndviNoThr[:, 0, 0])), 0)
        self.assertEqual(np.count_nonzero(np.isnan(ndviNoThr[:, 0, 232])), 3)
        self.assertEqual(np.count_nonzero(np.isnan(ndvi[:, 0, 0])), 0)
        self.assertEqual(np.count_nonzero(np.isnan(ndvi[:, 0, 232])), 3)
        
        # Test reading ndvi.
        ndviRead, ndviXref = mm.getNdvi()
        self.assertTrue(np.array_equal(ndvi, ndviRead, equal_nan=True))
        
    # -------------------------------------------------------------------------
    # testSortByNdvi
    # -------------------------------------------------------------------------
    def testSortByNdvi(self):

        mm = self.mmMod44
        band, bXref = mm.getBandCube(ProductTypeMod44.BAND3)
        bSorted: np.ndarray = np.sort(band, axis=0)
        ndvi, ndviXref = mm.getNdvi()
        nSorted = np.sort(ndvi, axis=0)

        # Test NDVI sort.  No-data values should be at the beginning.
        ndviSortedBand: np.ndarray = mm._sortByNDVI(band)
        self.assertFalse((ndviSortedBand == band).all())
        
        # Case 1:  band with no NaNs and NDVI with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(nSorted[11, :, :]))[0]
                           
        maxIndex = np.nanargmax(ndvi[:, x, y])
        self.assertEqual(maxIndex, 1)
        self.assertAlmostEqual(ndvi[maxIndex, x, y], 332.27445997)
        self.assertEqual(ndviSortedBand[-1, x, y], band[maxIndex, x, y])

        # ---
        # Case 2:  band with no NaNs and NDVI with some NaNs
        # This case does not exist.
        # x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
        #                    np.isnan(nSorted[11, :, :]))[0]
        # ---

        # ---
        # Case 3:  band with some NaNs and NDVI with no NaNs
        # This case does not exist.
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

        maxIndex = np.nanargmax(ndvi[:, x, y])
        self.assertEqual(maxIndex, 2)
        self.assertAlmostEqual(ndvi[maxIndex, x, y], 191.26912691)
        self.assertEqual(ndviSortedBand[-1, x, y], band[maxIndex, x, y])
        self.assertTrue(np.isnan(ndviSortedBand[0, x, y]))

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertTrue(np.isnan(ndviSortedBand[:, x, y]).all())

        # Case 6:  ndvi all NaN
        x, y = np.argwhere(np.isnan(nSorted[0, :, :]))[0]
        self.assertTrue(np.isnan(ndviSortedBand[:, x, y]).all())

        # ---
        # Ensure sorted values are NaN where NDVI values are NaN.  Find a case
        # where NDVI is NaN and band 1 is not.  Unable to find this case.
        # d, r, c = np.argwhere(np.isnan(ndvi) & ~np.isnan(band))
        # ---

        # ---
        # Test sorted by NDVI with no-data values at the end.
        # ---
        ndviSortedBand: np.ndarray = mm._sortByNDVI(band, noDataLow=False)
        
        # Case 1:  band with no NaNs and NDVI with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(nSorted[11, :, :]))[0]
                           
        maxIndex = np.nanargmax(ndvi[:, x, y])
        self.assertEqual(maxIndex, 1)
        self.assertAlmostEqual(ndvi[maxIndex, x, y], 332.27445997)
        self.assertEqual(ndviSortedBand[-1, x, y], band[maxIndex, x, y])

        # Case 4:  band with some NaNs and NDVI with some NaNs
        x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
                           ~np.isnan(bSorted[1, :, :]) &
                           np.isnan(bSorted[11, :, :]) &
                           np.isnan(nSorted[11, :, :]))[0]

        maxIndex = np.nanargmax(ndvi[:, x, y])
        self.assertEqual(maxIndex, 2)
        self.assertAlmostEqual(ndvi[maxIndex, x, y], 191.26912691)
        numNan = np.isnan(ndvi[:, x, y]).sum()
        
        self.assertEqual(ndviSortedBand[-numNan - 1, x, y], 
                         band[maxIndex, x, y])
        
        self.assertTrue(np.isnan(ndviSortedBand[-1, x, y]))

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertTrue(np.isnan(ndviSortedBand[:, x, y]).all())

        # Case 6:  ndvi all NaN
        x, y = np.argwhere(np.isnan(nSorted[0, :, :]))[0]
        self.assertTrue(np.isnan(ndviSortedBand[:, x, y]).all())

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

        # Case 1:  band with no NaNs and thermal with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(tSorted[11, :, :]))[0]
                           
        maxIndex = np.nanargmax(thermal[:, x, y])
        self.assertEqual(tSortedBand[-1, x, y], band[maxIndex, x, y])
        
        # ---
        # Case 2:  band with no NaNs and thermal with some NaNs
        # This case does not exist.
        # x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
        #                    np.isnan(tSorted[11, :, :]))[0]
        # ---

        # ---
        # Case 3:  band with some NaNs and thermal with no NaNs
        # This case does not exist.
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

        maxIndex = np.nanargmax(thermal[:, x, y])
        self.assertEqual(tSortedBand[-1, x, y], band[maxIndex, x, y])

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertTrue(np.isnan(tSortedBand[:, x, y]).all())

        # Case 6:  ndvi all NaN
        x, y = np.argwhere(np.isnan(tSorted[0, :, :]))[0]
        self.assertTrue(np.isnan(tSortedBand[:, x, y]).all())

        # ---
        # Test with no-data values at the end.
        # ---
        tSortedBand: np.ndarray = mm._sortByThermal(band, noDataLow=False)
        self.assertFalse((tSortedBand == band).all())

        thermal, tXref = mm.getBandCube(ProductTypeMod44.BAND31)
        tSorted = np.sort(thermal, axis=0)

        # Case 1:  band with no NaNs and thermal with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(tSorted[11, :, :]))[0]
                           
        maxIndex = np.nanargmax(thermal[:, x, y])
        self.assertEqual(tSortedBand[-1, x, y], band[maxIndex, x, y])
        
        # ---
        # Case 2:  band with no NaNs and thermal with some NaNs
        # This case does not exist.
        # x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
        #                    np.isnan(tSorted[11, :, :]))[0]
        # ---

        # ---
        # Case 3:  band with some NaNs and thermal with no NaNs
        # This case does not exist.
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

        maxIndex = np.nanargmax(thermal[:, x, y])

        numNan = np.isnan(thermal[:, x, y]).sum()
        
        self.assertEqual(tSortedBand[-numNan - 1, x, y], 
                         band[maxIndex, x, y])

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertTrue(np.isnan(tSortedBand[:, x, y]).all())

        # Case 6:  ndvi all NaN
        x, y = np.argwhere(np.isnan(tSorted[0, :, :]))[0]
        self.assertTrue(np.isnan(tSortedBand[:, x, y]).all())

    # -------------------------------------------------------------------------
    # testUnsortedMonthlyBands
    # -------------------------------------------------------------------------
    def testUnsortedMonthlyBands(self):
        
        METRIC_TITLE = 'UnsortedMonthlyBands'
        METRIC_NAME = 'metric' + METRIC_TITLE
        mm = self.mmMod09A
        
        # ---
        # Write and review a tile with a moderate solar zenith.
        # ---
        tid = 'h09v05'
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (96, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)
        
        # Day 193, July 12, is summer and should have a decent sun angle.
        day = 193
        
        xrefKey = METRIC_TITLE + '-' + ProductTypeMod09A.BAND1 + '-Day-' + \
                  str(self.year2019) + str(day).zfill(3)

        index = metric.dayXref[xrefKey]

        cdf = CompositeDayFile(self.productTypeMod09A, 
                               tid, 
                               self.year2019, 
                               day, 
                               ProductTypeMod09A.BAND1,
                               mm._compDir,
                               mm._dayDir)

        raster = cdf.getRaster
        raster = np.where(np.isnan(raster), -10001, raster).astype(int)
        self.assertTrue(np.array_equal(metric.cube[index], raster))
        
        # Test some random coordinates.
        coords = self.getRandomCoords()
        
        for coord in coords:
            
            self.assertEqual(metric.cube[index, coord[0], coord[1]],
                             raster[coord[0], coord[1]])
        
        # ---
        # Write and review a tile with a significant solar zenith.
        # ---
        tid = 'h12v02'
        day = 289
        
        xrefKey = METRIC_TITLE + '-' + ProductTypeMod09A.BAND1 + '-Day-' + \
                  str(self.year2019) + str(day).zfill(3)

        mm = Metrics(tid, 
                     self.year2019, 
                     self.productTypeMod09A, 
                     self._mod09OutDir, 
                     self._logger)
        
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (96, 4800, 4800))
        self.assertEqual(metric.name, 'UnsortedMonthlyBands')

        index = metric.dayXref[xrefKey]

        cdf = CompositeDayFile(self.productTypeMod09A, 
                               tid, 
                               self.year2019, 
                               day, 
                               ProductTypeMod09A.BAND1,
                               mm._compDir,
                               mm._dayDir)

        raster = cdf.getRaster
        raster = np.where(np.isnan(raster), -10001, raster).astype(int)

        x = 2888
        y = 426
        self.assertTrue(np.array_equal(metric.cube[index], raster))

        for coord in coords:
            
            self.assertEqual(metric.cube[index, coord[0], coord[1]],
                             raster[coord[0], coord[1]])
        
    # -------------------------------------------------------------------------
    # testBandReflMax
    # -------------------------------------------------------------------------
    def testBandReflMax(self):
        
        METRIC_TITLE = 'BandReflMax'
        METRIC_NAME = 'metric' + METRIC_TITLE
        mm = self.mmMod44
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)
        
        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND6 
        bIndex = metric.dayXref[xrefKey]
        band, bXref = mm.getBandCube(ProductTypeMod44.BAND6)
        bSorted = np.sort(band, axis=0)
        
        # Case 1:  band with no NaNs 
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], int(band[:, x, y].max()))
        
        # Case 3:  band with some NaNs
        x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
                           ~np.isnan(bSorted[1, :, :]) &
                           np.isnan(bSorted[11, :, :]))[0]

        self.assertEqual(metric.cube[bIndex, x, y], 
                         int(np.nanmax(band[:, x, y])))

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)

        # Test some random coordinates.
        coords = self.getRandomCoords()
        
        for coord in coords:
            
            self.assertEqual(metric.cube[bIndex, coord[0], coord[1]],
                             int(np.nanmax(band[:,coord[0],coord[1]])))
        
    # -------------------------------------------------------------------------
    # testBandReflMedian
    # -------------------------------------------------------------------------
    def testBandReflMedian(self):
        
        METRIC_TITLE = 'BandReflMedian'
        METRIC_NAME = 'metric' + METRIC_TITLE
        mm = self.mmMod44
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)
        
        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND6 
        bIndex = metric.dayXref[xrefKey]
        band, bXref = mm.getBandCube(ProductTypeMod44.BAND6)
        bSorted = np.sort(band, axis=0)
        
        # Case 1:  band with no NaNs 
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]))[0]
        exp = int(np.median(band[:, x, y]))
        self.assertEqual(metric.cube[bIndex, x, y], exp)
        
        # Case 3:  band with some NaNs
        x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
                           ~np.isnan(bSorted[1, :, :]) &
                           np.isnan(bSorted[11, :, :]))[0]

        exp = int(np.nanmedian(band[:, x, y]))
        self.assertEqual(metric.cube[bIndex, x, y], exp)

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)

        # Test random coordinates.
        coords = self.getRandomCoords()

        for coord in coords:

            self.assertEqual(metric.cube[bIndex, coord[0], coord[1]],
                             int(np.nanmedian(band[:,coord[0],coord[1]])))
        
    # -------------------------------------------------------------------------
    # testBandReflMin
    # -------------------------------------------------------------------------
    def testBandReflMin(self):
        
        METRIC_TITLE = 'BandReflMin'
        METRIC_NAME = 'metric' + METRIC_TITLE
        mm = self.mmMod44
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)
        
        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND6 
        bIndex = metric.dayXref[xrefKey]
        band, bXref = mm.getBandCube(ProductTypeMod44.BAND6)
        bSorted = np.sort(band, axis=0)
        
        # Case 1:  band with no NaNs 
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], int(band[:, x, y].min()))
        
        # Case 3:  band with some NaNs
        x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
                           ~np.isnan(bSorted[1, :, :]) &
                           np.isnan(bSorted[11, :, :]))[0]

        self.assertEqual(metric.cube[bIndex, x, y], 
                         int(np.nanmin(band[:, x, y])))

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)

        # Test some random coordinates.
        coords = self.getRandomCoords()
        
        for coord in coords:
            
            self.assertEqual(metric.cube[bIndex, coord[0], coord[1]],
                             int(np.nanmin(band[:,coord[0],coord[1]])))
        
    # -------------------------------------------------------------------------
    # testBandReflMaxGreenness
    # -------------------------------------------------------------------------
    def testBandReflMaxGreenness(self):
        
        METRIC_TITLE = 'BandReflMaxGreenness'
        METRIC_NAME = 'metric' + METRIC_TITLE
        mm = self.mmMod44
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)

        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND2
        bIndex = metric.dayXref[xrefKey]
        band, bXref = mm.getBandCube(ProductTypeMod44.BAND2)
        bSorted = np.sort(band, axis=0)

        ndvi, nXref = mm.getNdvi()
        nSorted = np.sort(ndvi, axis=0)
        # bSortedByN = self.mmMod44._sortByNDVI(band)
        
        # Case 1:  band with no NaNs and thermal with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(nSorted[11, :, :]))[0]
 
        maxNdviIndex = np.argmax(ndvi[:, x, y])
        exp = int(band[maxNdviIndex, x, y])
        self.assertEqual(metric.cube[bIndex, x, y], exp)
        
        # ---
        # Case 2:  band with no NaNs and NDVI with some NaNs
        # Unable to find this case.
        # x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
        #                    np.isnan(nSorted[11, :, :]))[0]
        # ---
                           
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
        
        maxNdviIndex = np.nanargmax(ndvi[:, x, y])
        exp = int(band[maxNdviIndex, x, y])
        self.assertEqual(metric.cube[bIndex, x, y], exp)

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)
        
        # Case 6:  thermal all NaN
        x, y = np.argwhere(np.isnan(nSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)

    # -------------------------------------------------------------------------
    # testBandReflMedianGreenness
    # -------------------------------------------------------------------------
    def testBandReflMedianGreenness(self):
        
        METRIC_TITLE = 'BandReflMedianGreenness'
        METRIC_NAME = 'metric' + METRIC_TITLE
        mm = self.mmMod44
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)

        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND1
        bIndex = metric.dayXref[xrefKey]
        band, bXref = mm.getBandCube(ProductTypeMod44.BAND1)
        bSorted = np.sort(band, axis=0)

        ndvi, nXref = mm.getNdvi()
        nSorted = np.sort(ndvi, axis=0)
        bSortedByN = mm._sortByNDVI(band)
        
        # ---
        # Case 1:  band with no NaNs and NDVI with no NaNs
        # ---
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(nSorted[11, :, :]))[0]
        
        index = int(ndvi[:, x, y].shape[0] / 2)
        exp = int(bSortedByN[index, x, y])
        self.assertEqual(metric.cube[bIndex, x, y], exp)
        
        # ---
        # Case 2:  band with no NaNs and NDVI with some NaNs
        # Unable to find this case.
        # x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
        #                    np.isnan(nSorted[11, :, :]))[0]
        # ---
                           
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
        self.assertEqual(metric.cube[bIndex, x, y], exp)
        
        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)
        
        # Case 6:  thermal all NaN
        x, y = np.argwhere(np.isnan(nSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)
        
    # -------------------------------------------------------------------------
    # testBandReflMinGreenness
    # -------------------------------------------------------------------------
    def testBandReflMinGreenness(self):
        
        METRIC_TITLE = 'BandReflMinGreenness'
        METRIC_NAME = 'metric' + METRIC_TITLE
        mm = self.mmMod44
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)

        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND2
        bIndex = metric.dayXref[xrefKey]
        band, bXref = mm.getBandCube(ProductTypeMod44.BAND2)
        bSorted = np.sort(band, axis=0)

        ndvi, nXref = mm.getNdvi()
        nSorted = np.sort(ndvi, axis=0)
        
        # Case 1:  band with no NaNs and thermal with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(nSorted[11, :, :]))[0]
 
        minNdviIndex = np.argmin(ndvi[:, x, y])
        exp = int(band[minNdviIndex, x, y])
        self.assertEqual(metric.cube[bIndex, x, y], exp)
        
        # ---
        # Case 2:  band with no NaNs and NDVI with some NaNs
        # Unable to find this case.
        # x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
        #                    np.isnan(nSorted[11, :, :]))[0]
        # ---
                           
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
        
        minNdviIndex = np.nanargmin(ndvi[:, x, y])
        exp = int(band[minNdviIndex, x, y])
        self.assertEqual(metric.cube[bIndex, x, y], exp)

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)
        
        # Case 6:  thermal all NaN
        x, y = np.argwhere(np.isnan(nSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)

    # -------------------------------------------------------------------------
    # testBandReflMaxTemp
    # -------------------------------------------------------------------------
    def testBandReflMaxTemp(self):
        
        METRIC_TITLE = 'BandReflMaxTemp'
        METRIC_NAME = 'metric' + METRIC_TITLE
        mm = self.mmMod44
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)

        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND4
        bIndex = metric.dayXref[xrefKey]
        band, b1Xref = mm.getBandCube(ProductTypeMod44.BAND4)
        bSorted = np.sort(band, axis=0)

        thermal, tXref = mm.getBandCube(ProductTypeMod44.BAND31)
        tSorted = np.sort(thermal, axis=0)

        # ---
        # Case 1:  band with no NaNs and thermal with no NaNs
        # ---
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(tSorted[11, :, :]))[0]
        
        maxIndex = np.argmax(thermal[:, x, y])
        self.assertEqual(metric.cube[bIndex, x, y], band[maxIndex, x, y])

        # ---
        # Case 2:  band with no NaNs and NDVI with some NaNs
        # Unable to find this case.
        # x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
        #                    np.isnan(tSorted[11, :, :]))[0]
        # ---

        # ---
        # Case 3:  band with some NaNs and NDVI with no NaNs
        # Unable to find this case.
        # x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
        #                    ~np.isnan(bSorted[1, :, :]) &
        #                    np.isnan(bSorted[11, :, :]) &
        #                    ~np.isnan(tSorted[11, :, :]))[0]
        # ---

        # Case 4:  band with some NaNs and NDVI with some NaNs
        x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
                           ~np.isnan(bSorted[1, :, :]) &
                           np.isnan(bSorted[11, :, :]) &
                           np.isnan(tSorted[11, :, :]))[0]
        
        maxIndex = np.nanargmax(thermal[:, x, y])
        self.assertEqual(metric.cube[bIndex, x, y], band[maxIndex, x, y])

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)
        
        # Case 6:  thermal all NaN
        x, y = np.argwhere(np.isnan(tSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)
        
    # -------------------------------------------------------------------------
    # testBandReflMedianTemp
    # -------------------------------------------------------------------------
    def testBandReflMedianTemp(self):
        
        METRIC_TITLE = 'BandReflMedianTemp'
        METRIC_NAME = 'metric' + METRIC_TITLE
        mm = self.mmMod44
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)

        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND4
        bIndex = metric.dayXref[xrefKey]
        band, bXref = mm.getBandCube(ProductTypeMod44.BAND4)
        bSorted = np.sort(band, axis=0)

        thermal, tXref = mm.getBandCube(ProductTypeMod44.BAND31)
        tSorted = np.sort(thermal, axis=0)
        bSortedByT = mm._sortByThermal(band)
        
        # Case 1:  band with no NaNs and thermal with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(tSorted[11, :, :]))[0]

        index = int(12 / 2)
        exp = int(bSortedByT[index, x, y])
        self.assertEqual(metric.cube[bIndex, x, y], exp)
        
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
        self.assertEqual(metric.cube[bIndex, x, y], exp)
        
        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)
        
        # Case 6:  thermal all NaN
        x, y = np.argwhere(np.isnan(tSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)

    # -------------------------------------------------------------------------
    # testBandReflMinTemp
    # -------------------------------------------------------------------------
    def testBandReflMinTemp(self):
        
        METRIC_TITLE = 'BandReflMinTemp'
        METRIC_NAME = 'metric' + METRIC_TITLE
        mm = self.mmMod44
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)

        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND4
        bIndex = metric.dayXref[xrefKey]
        band, b1Xref = mm.getBandCube(ProductTypeMod44.BAND4)
        bSorted = np.sort(band, axis=0)

        thermal, tXref = mm.getBandCube(ProductTypeMod44.BAND31)
        tSorted = np.sort(thermal, axis=0)

        # ---
        # Case 1:  band with no NaNs and thermal with no NaNs
        # ---
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(tSorted[11, :, :]))[0]
        
        minIndex = np.argmin(thermal[:, x, y])
        self.assertEqual(metric.cube[bIndex, x, y], band[minIndex, x, y])

        # ---
        # Case 2:  band with no NaNs and NDVI with some NaNs
        # Unable to find this case.
        # x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
        #                    np.isnan(tSorted[11, :, :]))[0]
        # ---

        # ---
        # Case 3:  band with some NaNs and NDVI with no NaNs
        # Unable to find this case.
        # x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
        #                    ~np.isnan(bSorted[1, :, :]) &
        #                    np.isnan(bSorted[11, :, :]) &
        #                    ~np.isnan(tSorted[11, :, :]))[0]
        # ---

        # Case 4:  band with some NaNs and NDVI with some NaNs
        x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
                           ~np.isnan(bSorted[1, :, :]) &
                           np.isnan(bSorted[11, :, :]) &
                           np.isnan(tSorted[11, :, :]))[0]
        
        minIndex = np.nanargmin(thermal[:, x, y])
        self.assertEqual(metric.cube[bIndex, x, y], band[minIndex, x, y])

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)
        
        # Case 6:  thermal all NaN
        x, y = np.argwhere(np.isnan(tSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod44.NO_DATA)
        
    # -------------------------------------------------------------------------
    # testLowestMeanBandRefl
    # -------------------------------------------------------------------------
    def _testLowestMeanBandRefl(self, numLowest: int):
        
        METRIC_TITLE = 'Lowest' + str(numLowest) + 'MeanBandRefl'
        METRIC_NAME = 'metric' + METRIC_TITLE
        mm = self.mmMod44
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (7, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)

        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND1
        bIndex = metric.dayXref[xrefKey]
        band, bXref = mm.getBandCube(ProductTypeMod44.BAND1)
        bSorted = np.sort(band, axis=0)
        
        # Case 1:  band with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]))[0]
        exp = int(np.mean(bSorted[:numLowest, x, y]))
        self.assertEqual(metric.cube[bIndex, x, y], exp)
        
        # Case 2:  band with some NaNs
        x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
                           ~np.isnan(bSorted[1, :, :]) &
                           np.isnan(bSorted[11, :, :]))[0]

        exp = int(np.nanmean(bSorted[:numLowest, x, y]))
        self.assertEqual(metric.cube[bIndex, x, y], exp)
        
        # Case 3:  band with all NaNs
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        exp = ProductTypeMod44.NO_DATA
        self.assertEqual(metric.cube[bIndex, x, y], exp)
        
    # -------------------------------------------------------------------------
    # testLowest3MeanBandRefl
    # -------------------------------------------------------------------------
    def testLowest3MeanBandRefl(self):
        self._testLowestMeanBandRefl(3)
        
    # -------------------------------------------------------------------------
    # testLowest6MeanBandRefl
    # -------------------------------------------------------------------------
    def testLowest6MeanBandRefl(self):
        self._testLowestMeanBandRefl(6)
        
    # -------------------------------------------------------------------------
    # testLowest8MeanBandRefl
    # -------------------------------------------------------------------------
    def testLowest8MeanBandRefl(self):
        self._testLowestMeanBandRefl(8)
        
    # -------------------------------------------------------------------------
    # testGreenestMeanBandRefl
    # -------------------------------------------------------------------------
    def _testGreenestMeanBandRefl(self, numGreenest: int):
        
        mm = self.mmMod09A
        
        # Run the metric.
        METRIC_TITLE = 'Greenest' + str(numGreenest) + 'MeanBandRefl'
        METRIC_NAME = 'metric' + METRIC_TITLE
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)
        
        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND3
        bIndex = metric.dayXref[xrefKey]
        band, bXref = mm.getBandCube(ProductTypeMod44.BAND3)
        bSorted = np.sort(band, axis=0)

        ndvi, nXref = mm.getNdvi()
        nSorted = np.sort(ndvi, axis=0)
        bSortedByN = mm._sortByNDVI(band)

        # Case 1:  band with no NaNs and NDVI with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(ndvi[11, :, :]))[0]

        bSliced = bSortedByN[-numGreenest:, x, y]
        exp = int(np.nanmean(bSliced))
        self.assertEqual(metric.cube[bIndex, x, y], exp)
        
        # ---
        # Case 2:  band with no NaNs and NDVI with some NaNs
        # There are no locations where all 12 months are not NaN.
        # x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
        #                    np.isnan(nSorted[11, :, :]))[0]
        # ---

        # ---
        # Case 3:  band with some NaNs and NDVI with no NaNs
        # There are no locations where all 12 months of NDVI are not NaN.
        # x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
        #                    ~np.isnan(bSorted[1, :, :]) &
        #                    np.isnan(bSorted[11, :, :]) &
        #                    ~np.isnan(nSorted[11, :, :]))[0]
        # ---

        # Case 4:  band with some NaNs and NDVI with some NaNs (2894, 708)
        x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
                           ~np.isnan(bSorted[1, :, :]) &
                           np.isnan(bSorted[11, :, :]) &
                           np.isnan(nSorted[11, :, :]))[0]

        bSliced = bSortedByN[-numGreenest:, x, y]
        exp = int(np.nanmean(bSliced))
        self.assertEqual(metric.cube[bIndex, x, y], exp)
        
        # Case 5:  band all NaN (0, 0)
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod09A.NO_DATA)
        
        # Case 6:  ndvi all NaN (0, 0)
        x, y = np.argwhere(np.isnan(nSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[bIndex, x, y], ProductTypeMod09A.NO_DATA)

    # -------------------------------------------------------------------------
    # testGreenest3MeanBandRefl
    # -------------------------------------------------------------------------
    def testGreenest3MeanBandRefl(self):
        self._testGreenestMeanBandRefl(3)
        
    # -------------------------------------------------------------------------
    # testGreenest6MeanBandRefl
    # -------------------------------------------------------------------------
    def testGreenest6MeanBandRefl(self):
        self._testGreenestMeanBandRefl(6)
        
    # -------------------------------------------------------------------------
    # testGreenest8MeanBandRefl
    # -------------------------------------------------------------------------
    def testGreenest8MeanBandRefl(self):
        self._testGreenestMeanBandRefl(8)
        
    # -------------------------------------------------------------------------
    # testWarmestMeanBandRefl
    # -------------------------------------------------------------------------
    def _testWarmestMeanBandRefl(self, numWarmest: int):
        
        mm = self.mmMod44
        METRIC_TITLE = 'Warmest' + str(numWarmest) + 'MeanBandRefl'
        METRIC_NAME = 'metric' + METRIC_TITLE
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)

        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND3
        bIndex = metric.dayXref[xrefKey]
        band, b1Xref = mm.getBandCube(ProductTypeMod44.BAND3)
        bSorted = np.sort(band, axis=0)
        
        thermal, tXref = mm.getBandCube(ProductTypeMod44.BAND31)
        tSorted = np.sort(thermal, axis=0)
        bSortedByT = mm._sortByThermal(band)
        
        # Case 1:  band with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]))[0]
        exp = int(np.nanmean(bSortedByT[-numWarmest:, x, y]))
        self.assertEqual(metric.cube[bIndex, x, y], exp)
        
        # ---
        # Case 2:  band with no NaNs and thermal with some NaNs
        # This case does not exist.
        # x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
        #                    np.isnan(tSorted[11, :, :]))[0]
        # ---

        # ---
        # Case 3:  band with some NaNs and thermal with no NaNs
        # This case does not exist.
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

        exp = int(np.nanmean(bSortedByT[-numWarmest:, x, y]))
        self.assertEqual(metric.cube[bIndex, x, y], exp)

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        exp = ProductTypeMod44.NO_DATA
        self.assertEqual(metric.cube[bIndex, x, y], exp)

        # Case 6:  thermal all NaN
        x, y = np.argwhere(np.isnan(tSorted[0, :, :]))[0]
        exp = ProductTypeMod44.NO_DATA
        self.assertEqual(metric.cube[bIndex, x, y], exp)

    # -------------------------------------------------------------------------
    # testWarmest3MeanBandRefl
    # -------------------------------------------------------------------------
    def testWarmest3MeanBandRefl(self):
        self._testWarmestMeanBandRefl(3)
        
    # -------------------------------------------------------------------------
    # testWarmest6MeanBandRefl
    # -------------------------------------------------------------------------
    def testWarmest6MeanBandRefl(self):
        self._testWarmestMeanBandRefl(6)
        
    # -------------------------------------------------------------------------
    # testWarmest8MeanBandRefl
    # -------------------------------------------------------------------------
    def testWarmest8MeanBandRefl(self):
        self._testWarmestMeanBandRefl(8)
        
    # -------------------------------------------------------------------------
    # testAmpBandRefl
    # -------------------------------------------------------------------------
    def testAmpBandRefl(self):
        
        mm = self.mmMod44
        METRIC_TITLE = 'AmpBandRefl'
        METRIC_NAME = 'metric' + METRIC_TITLE
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)

        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND6 
        index = metric.dayXref[xrefKey]

        band, bXref = mm.getBandCube(ProductTypeMod44.BAND6)
        bSorted = np.sort(band, axis=0)

        # Case 1:  band with no NaNs 
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]))[0]
        exp = int(band[:, x, y].max() - band[:, x, y].min())
        self.assertEqual(metric.cube[index, x, y], exp)
        
        # Case 2:  band with some NaNs
        x, y = np.argwhere(~np.isnan(bSorted[0, :, :]) &
                           ~np.isnan(bSorted[1, :, :]) &
                           np.isnan(bSorted[11, :, :]))[0]

        exp = int(np.nanmax(band[:, x, y]) - np.nanmin(band[:, x, y]))
        self.assertEqual(metric.cube[index, x, y], exp)

        # Case 3:  band with all NaNs
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        exp = ProductTypeMod44.NO_DATA
        self.assertEqual(metric.cube[index, x, y], exp)

    # -------------------------------------------------------------------------
    # testAmpGreenestBandRefl
    # -------------------------------------------------------------------------
    def testAmpGreenestBandRefl(self):

        mm = self.mmMod44
        METRIC_TITLE = 'AmpGreenestBandRefl'
        METRIC_NAME = 'metric' + METRIC_TITLE
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)

        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND4 
        index = metric.dayXref[xrefKey]

        band, bXref = mm.getBandCube(ProductTypeMod44.BAND4)
        bSorted = np.sort(band, axis=0)

        ndvi, nXref = mm.getNdvi()
        nSorted = np.sort(ndvi, axis=0)
        # bSortedByN = mm._sortByNDVI(band)
        
        # Case 1:  band with no NaNs and NDVI with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(nSorted[11, :, :]))[0]

        minIndex = np.argmin(ndvi[:, x, y])
        maxIndex = np.argmax(ndvi[:, x, y])
        exp = abs(int(band[maxIndex, x, y] - band[minIndex, x, y]))
        self.assertEqual(metric.cube[index, x, y], exp)

        # ---
        # Case 2:  band with no NaNs and NDVI with some NaNs
        # This case does not exist.
        # x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
        #                    np.isnan(nSorted[11, :, :]))[0]
        # ---

        # ---
        # Case 3:  band with some NaNs and NDVI with no NaNs
        # This case does not exist.
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

        minIndex = np.nanargmin(ndvi[:, x, y])
        maxIndex = np.nanargmax(ndvi[:, x, y])
        exp = abs(int(band[maxIndex, x, y] - band[minIndex, x, y]))
        self.assertEqual(metric.cube[index, x, y], exp)

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        exp = ProductTypeMod44.NO_DATA
        self.assertEqual(metric.cube[index, x, y], exp)

        # Case 6:  ndvi all NaN
        x, y = np.argwhere(np.isnan(nSorted[0, :, :]))[0]
        exp = ProductTypeMod44.NO_DATA
        self.assertEqual(metric.cube[index, x, y], exp)
        
    # -------------------------------------------------------------------------
    # testAmpWarmestBandRefl
    # -------------------------------------------------------------------------
    def testAmpWarmestBandRefl(self):

        mm = self.mmMod44
        METRIC_TITLE = 'AmpWarmestBandRefl'
        METRIC_NAME = 'metric' + METRIC_TITLE
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)

        xrefKey = METRIC_TITLE + '-' + ProductTypeMod44.BAND5 
        index = metric.dayXref[xrefKey]

        band, bXref = mm.getBandCube(ProductTypeMod44.BAND5)
        bSorted = np.sort(band, axis=0)

        thermal, tXref = mm.getBandCube(ProductTypeMod44.BAND31)
        tSorted = np.sort(thermal, axis=0)
        # bSortedByT = mm._sortByThermal(band)
        
        # Case 1:  band with no NaNs and thermal with no NaNs
        x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
                           ~np.isnan(tSorted[11, :, :]))[0]

        minIndex = np.nanargmin(thermal[:, x, y])
        maxIndex = np.nanargmax(thermal[:, x, y])
        exp = abs(int(band[maxIndex, x, y] - band[minIndex, x, y]))
        self.assertEqual(metric.cube[index, x, y], exp)

        # ---
        # Case 2:  band with no NaNs and thermal with some NaNs
        # This case does not exist.
        # x, y = np.argwhere(~np.isnan(bSorted[11, :, :]) &
        #                    np.isnan(tSorted[11, :, :]))[0]
        # ---

        # ---
        # Case 3:  band with some NaNs and thermal with no NaNs
        # This case does not exist.
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

        minIndex = np.nanargmin(thermal[:, x, y])
        maxIndex = np.nanargmax(thermal[:, x, y])
        exp = abs(int(band[maxIndex, x, y] - band[minIndex, x, y]))
        self.assertEqual(metric.cube[index, x, y], exp)

        # Case 5:  band all NaN
        x, y = np.argwhere(np.isnan(bSorted[0, :, :]))[0]
        exp = ProductTypeMod44.NO_DATA
        self.assertEqual(metric.cube[index, x, y], exp)

        # Case 6:  thermal all NaN
        x, y = np.argwhere(np.isnan(tSorted[0, :, :]))[0]
        exp = ProductTypeMod44.NO_DATA
        self.assertEqual(metric.cube[index, x, y], exp)

    # -------------------------------------------------------------------------
    # testTempMeanWarmest3
    # -------------------------------------------------------------------------
    def testTempMeanWarmest3(self):

        mm = self.mmMod44
        METRIC_TITLE = 'TempMeanWarmest3'
        METRIC_NAME = 'metric' + METRIC_TITLE
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (1, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)

        thermal, tXref = mm.getBandCube(ProductTypeMod44.BAND31)
        tSorted = np.sort(thermal, axis=0)
        
        # Case 1:  thermal with no NaNs
        x, y = np.argwhere(~np.isnan(tSorted[11, :, :]))[0]
        exp = int(np.mean(tSorted[-3:, x, y], axis=0))
        self.assertEqual(metric.cube[0, x, y], exp)

        # Case 2:  thermal with some NaNs
        x, y = np.argwhere(~np.isnan(tSorted[0, :, :]) &
                           np.isnan(tSorted[11, :, :]))[0]

        numNan = np.isnan(tSorted[:, x, y]).sum()
        exp = int(np.mean(tSorted[-numNan-3:-numNan, x, y]))
        self.assertEqual(metric.cube[0, x, y], exp)
        
        # Case 3:  thermal all NaN
        x, y = np.argwhere(np.isnan(tSorted[0, :, :]))[0]
        self.assertEqual(metric.cube[0, x, y], ProductTypeMod44.NO_DATA)
        
        # This point was a problem.
        x = 138
        y = 1158
        self.assertFalse(np.isnan(metric.cube[0, x, y]))
        
    # -------------------------------------------------------------------------
    # testTempMeanGreenest3
    # -------------------------------------------------------------------------
    def testTempMeanGreenest3(self):

        mm = self.mmMod44
        METRIC_TITLE = 'TempMeanGreenest3'
        METRIC_NAME = 'metric' + METRIC_TITLE
        metName = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        metric: Band = mm.getMetric(METRIC_NAME)
        self.assertEqual(metric.cube.shape, (1, 4800, 4800))
        self.assertEqual(metric.name, METRIC_TITLE)

        ndvi, ndviXref = mm.getNdvi()
        tSorted = np.sort(ndvi, axis=0)
        
        # Start with a location with no NaNs.
        x, y = np.argwhere(~np.isnan(tSorted[11, :, :]))[0]
        
        mean = ((tSorted[9, x, y] + \
                 tSorted[10, x, y] + \
                 tSorted[11, x, y]) / 3.0).astype(int)
        
        self.assertEqual(metric.cube[0, x, y], mean)

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

        self.assertEqual(metric.cube[0, x, y], mean)

        # Test the last two (greatest) thermal values being NaN.
        x, y = np.argwhere(~np.isnan(tSorted[0, :, :]) &
                           ~np.isnan(tSorted[1, :, :]) &
                           np.isnan(tSorted[10, :, :]))[0]

        self.assertTrue(np.isnan(tSorted[-2, x, y]))

        mean = ((tSorted[7, x, y] + tSorted[8, x, y] + tSorted[9, x, y]) / \
                3.0).astype(int)

        self.assertEqual(metric.cube[0, x, y], mean)

        # Test the last three (greatest) thermal values being NaN.
        x, y = np.argwhere(~np.isnan(tSorted[0, :, :]) &
                           ~np.isnan(tSorted[1, :, :]) &
                           np.isnan(tSorted[9, :, :]))[0]

        self.assertTrue(np.isnan(tSorted[-3, x, y]))

        mean = ((tSorted[6, x, y] + tSorted[7, x, y] + tSorted[8, x, y]) / \
                3.0).astype(int)

        self.assertEqual(metric.cube[0, x, y], mean)

        # Test all NaN.
        x, y = np.argwhere(np.isnan(tSorted[0, :, :]))[0]
        self.assertTrue(np.isnan(tSorted[0, x, y]))
        self.assertTrue(np.isnan(tSorted[1, x, y]))
        self.assertTrue(np.isnan(tSorted[-1, x, y]))
        self.assertEqual(metric.cube[0, x, y], ProductTypeMod44.NO_DATA)

    # -------------------------------------------------------------------------
    # marksTest
    # -------------------------------------------------------------------------
    def marksTest(self):
        
        # 1. Run them normally.
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/allQa')

        # 2. Run them with cloud-mixed off.
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/cloudMixedOff')

        # 3. Run them with cloud-shadow off.
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/cloudShadowOff')

        # 4. Run them with cloudy off.
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/cloudyOff')

        # 5. Run them with aerosol climatology off.
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/aeroClimOff')

        # 6. Run them with aerosol climatology and cloud-mixed off.
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/aeroCloudMixedOff')

        # 7. Run them with aerosol climatology and cloud-shadow off.
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/aeroCloudShadowOff')

        # 8. Run them with aerosol climatology and cloudy off.
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/aeroCloudyOff')

        # 9. Run them with aerosol climatology and cloud-internal off.
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/aeroCloudInternalOff')

        # ---
        # 10. All QA, threshold off
        # h12v02: all no-data values
        # ---
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/allQa-threshOff')

        # ---
        # 11. Aero off, threshold off
        # h12v02: mostly no-data values
        # ---
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/aeroOff-threshOff')

        # ---
        # 12. Aero off, threshold off
        # h09v05: looks good
        # ---
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/aeroOff-threshOff')

        # ---
        # 13. All QA, threshold off
        # h09v05: looks good, but more no-data values than #12
        # ---
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/allQa-threshOff')

        # ---
        # 14. All QA, threshold off
        # h12v09: looks ok, but a moderate amount of no-data values throughout
        # ---
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/allQa-threshOff')

        # ---
        # 15. All QA, threshold off
        # h12v09: looks ok, but a moderate amount of no-data values throughout
        # ---
        # outDir = Path('/explore/nobackup/people/rlgill' +
        #               '/SystemTesting/modis-vcf/MOD09A/aeroOff-threshOff')

        # Run the metrics.
        # tid = 'h12v09'
        # METRIC_NAME = 'metricGreenest3MeanBandRefl'
        # mm = Metrics(tid, julYear, pt, outDir, MetricsTestCase._logger)
        # metric: Band = mm.getMetric(METRIC_NAME, applyThreshold=False)
        # self.assertEqual(metric.cube.shape, (8, 4800, 4800))
        # self.assertEqual(metric.name, 'Greenest3MeanBandRefl')

        tid = 'h12v02'
        
        mm = Metrics(tid,
                     self.year2019,
                     self.productTypeMod09A,
                     self._mod09OutDir,
                     self._logger)

        METRIC_TITLE = 'AmpWarmestBandRefl'
        METRIC_NAME = 'metric' + METRIC_TITLE
        metName: Band = mm._metricsDir / (METRIC_TITLE + '.tif')
        metName.unlink(missing_ok=True)
        import pdb
        pdb.set_trace()
        # metric = mm.getMetric(METRIC_NAME)

        # Mark's points should not be no-data values
        x = 158
        y = 1138
        
        # More questionable no-data values
        x = 2256  # col
        y = 795   # row

        # Amp = 0
        x = 2538  # col
        y = 769   # row
        
        inDir = Path('/explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles/MOD09A/h12v02/2019/1-Days')
        
        files = [Path('MOD44-h12v02-2019065-Band31.bin'),
                 Path('MOD44-h12v02-2019081-Band31.bin'),
                 Path('MOD44-h12v02-2019097-Band31.bin'),
                 Path('MOD44-h12v02-2019113-Band31.bin'),
                 Path('MOD44-h12v02-2019129-Band31.bin'),
                 Path('MOD44-h12v02-2019145-Band31.bin'),
                 Path('MOD44-h12v02-2019161-Band31.bin'),
                 Path('MOD44-h12v02-2019177-Band31.bin'),
                 Path('MOD44-h12v02-2019193-Band31.bin'),
                 Path('MOD44-h12v02-2019209-Band31.bin'),
                 Path('MOD44-h12v02-2019225-Band31.bin'),
                 Path('MOD44-h12v02-2019241-Band31.bin'),
                 Path('MOD44-h12v02-2019257-Band31.bin'),
                 Path('MOD44-h12v02-2019273-Band31.bin'),
                 Path('MOD44-h12v02-2019289-Band31.bin'),
                 Path('MOD44-h12v02-2019305-Band31.bin'),
                 Path('MOD44-h12v02-2019321-Band31.bin'),
                 Path('MOD44-h12v02-2019337-Band31.bin'),
                 Path('MOD44-h12v02-2019353-Band31.bin'),
                 Path('MOD44-h12v02-2020001-Band31.bin'),
                 Path('MOD44-h12v02-2020017-Band31.bin'),
                 Path('MOD44-h12v02-2020033-Band31.bin'),
                 Path('MOD44-h12v02-2020049-Band31.bin')]
        
        for f in files:
            
            outBand = np.fromfile(inDir / f, dtype=np.int16). \
                      reshape(self.productTypeMod09A.ROWS,
                              self.productTypeMod09A.COLS)
                      
            print(f.name, outBand[x,y])
            
