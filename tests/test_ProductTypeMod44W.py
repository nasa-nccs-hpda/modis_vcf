
from pathlib import Path
import unittest

import numpy as np

from osgeo import gdal

from modis_vcf.model.ProductType import ProductType
from modis_vcf.model.ProductTypeMod44W import ProductTypeMod44W


# -----------------------------------------------------------------------------
# class ProductTypeMod44WTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_ProductTypeMod44W
# python -m unittest modis_vcf.model.tests.test_ProductTypeMod44W.ProductTypeMod44WTestCase.testInit
# -----------------------------------------------------------------------------
class ProductTypeMod44WTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUpClass
    # -------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):

        cls._inDir = Path('/css/modis/Collection6.1/L3/MOD44W-LandWaterMask')

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        pt = ProductTypeMod44W(ProductTypeMod44WTestCase._inDir)
        self.assertEqual(pt.inputDir, ProductTypeMod44WTestCase._inDir)
        self.assertEqual(pt.productType, 'MOD44')

    # -------------------------------------------------------------------------
    # testDayStep
    # -------------------------------------------------------------------------
    def testDayStep(self):

        pt = ProductTypeMod44W(ProductTypeMod44WTestCase._inDir)
        self.assertEqual(pt.dayStep, ProductTypeMod44W.DAY_STEP)
        
    # -------------------------------------------------------------------------
    # testFindFile
    # -------------------------------------------------------------------------
    def testFindFile(self):
        
        pt = ProductTypeMod44W(ProductTypeMod44WTestCase._inDir)
        
        tid = 'h09v05'
        year = 2019
        day = 1
        bandName = ProductTypeMod44W.WATER_MASK
        fileName: Path = pt.findFile(tid, year, day, bandName)
        
        expFile = '/css/modis/Collection6.1/L3/' + \
                  'MOD44W-LandWaterMask/2019/001/' + \
                  'MOD44W.A2019001.h09v05.061.2024007231940.hdf'
                  
        self.assertEqual(str(fileName), expFile)

        # Test a file that should not exist.
        day = 2112

        with self.assertRaisesRegex(RuntimeError, 'Unable to find file for'):
            fileName: Path = pt.findFile(tid, year, day, bandName)
        
    # -------------------------------------------------------------------------
    # testInputDir
    # -------------------------------------------------------------------------
    def testInputDir(self):
        
        pt = ProductTypeMod44W(ProductTypeMod44WTestCase._inDir)
        self.assertEqual(pt.inputDir, ProductTypeMod44WTestCase._inDir)

    # -------------------------------------------------------------------------
    # testSolarZenithScaleFactor
    # -------------------------------------------------------------------------
    # def testSolarZenithScaleFactor(self):
    #
    #     pt = ProductTypeMod44W(ProductTypeMod44WTestCase._inDir)
    #     self.assertEqual(pt.solarZenithScaleFactor, 1.0)
        
    # -------------------------------------------------------------------------
    # testYearOneDays
    # -------------------------------------------------------------------------
    def testYearOneDays(self):

        pt = ProductTypeMod44W(ProductTypeMod44WTestCase._inDir)
        expDays = [1]
        self.assertEqual(expDays, pt.yearOneDays)
        
    # -------------------------------------------------------------------------
    # testYearTwoDays
    # -------------------------------------------------------------------------
    def testYearTwoDays(self):

        pt = ProductTypeMod44W(ProductTypeMod44WTestCase._inDir)
        expDays = []
        self.assertEqual(expDays, pt.yearTwoDays)
        
    # -------------------------------------------------------------------------
    # testYearOneStartDay
    # -------------------------------------------------------------------------
    def testYearOneStartDay(self):

        pt = ProductTypeMod44W(ProductTypeMod44WTestCase._inDir)

        self.assertEqual(pt.yearOneStartDay,
                         ProductTypeMod44W.YEAR_ONE_START_DAY)

    # -------------------------------------------------------------------------
    # testYearOneEndDay
    # -------------------------------------------------------------------------
    def testYearOneEndDay(self):

        pt = ProductTypeMod44W(ProductTypeMod44WTestCase._inDir)
        self.assertEqual(pt.yearOneEndDay, ProductTypeMod44W.YEAR_ONE_END_DAY)

    # -------------------------------------------------------------------------
    # testYearTwoStartDay
    # -------------------------------------------------------------------------
    def testYearTwoStartDay(self):

        pt = ProductTypeMod44W(ProductTypeMod44WTestCase._inDir)

        self.assertEqual(pt.yearTwoStartDay,
                         ProductTypeMod44W.YEAR_TWO_START_DAY)

    # -------------------------------------------------------------------------
    # testYearTwoEndDay
    # -------------------------------------------------------------------------
    def testYearTwoEndDay(self):

        pt = ProductTypeMod44W(ProductTypeMod44WTestCase._inDir)
        self.assertEqual(pt.yearTwoEndDay, ProductTypeMod44W.YEAR_TWO_END_DAY)

    # -------------------------------------------------------------------------
    # testBandXref
    # -------------------------------------------------------------------------
    def testBandXref(self):

        pt = ProductTypeMod44W(ProductTypeMod44WTestCase._inDir)

        wName = pt.inputDir / ('2019/001/' + 
                               'MOD44W.A2019001.h09v05.061.2024007231940.hdf')

        wDs: gdal.Dataset = gdal.Open(str(wName))
        wSubs = wDs.GetSubDatasets()

        # Water mask
        subdatasetIndex: int = pt.bandXref[pt.WATER_MASK]
        desc: str = wSubs[subdatasetIndex][1]
        self.assertTrue('water_mask' in desc)

