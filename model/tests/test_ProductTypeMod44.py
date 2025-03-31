
from pathlib import Path
import unittest

import numpy as np

from osgeo import gdal

from modis_vcf.model.ProductType import ProductType
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44


# -----------------------------------------------------------------------------
# class ProductTypeMod44TestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_ProductTypeMod44
# python -m unittest modis_vcf.model.tests.test_ProductTypeMod44.ProductTypeMod44TestCase.testInit
# -----------------------------------------------------------------------------
class ProductTypeMod44TestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUpClass
    # -------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):

        cls._inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        pt = ProductTypeMod44(ProductTypeMod44TestCase._inDir)
        self.assertEqual(pt.inputDir, ProductTypeMod44TestCase._inDir)
        self.assertEqual(pt.productType, 'MOD44')

    # -------------------------------------------------------------------------
    # testCreateQaMask
    # -------------------------------------------------------------------------
    def testCreateQaMask(self):
        
        cqFile = ProductTypeMod44TestCase._inDir / \
                 'MOD44CQ.A2019065.h09v05.061.2020290183523.hdf'

        cqDs: gdal.Dataset = gdal.Open(str(cqFile))

        # Get the state, which does not use the no-data value.
        stateIndex = 0
        bandDs = gdal.Open(cqDs.GetSubDatasets()[stateIndex][0])
        
        state: np.ndarray = bandDs.ReadAsArray(buf_xsize=ProductType.COLS,
                                               buf_ysize=ProductType.ROWS)
                                               
        state = state.astype(np.int16)

        # Get the solar zenith, which uses the no-data value.
        solzIndex = 3
        bandDs = gdal.Open(cqDs.GetSubDatasets()[solzIndex][0])
        bandNoData = bandDs.GetRasterBand(1).GetNoDataValue()
        
        solz: np.ndarray = bandDs.ReadAsArray(buf_xsize=ProductType.COLS,
                                              buf_ysize=ProductType.ROWS)
                                              
        solz = np.where(solz == bandNoData, -10001, solz)
        
        # Create the mask.
        pt = ProductTypeMod44(ProductTypeMod44TestCase._inDir)
        qaMask: np.ndarray = pt.createQaMask(state, solz, 72)

    # -------------------------------------------------------------------------
    # testDayStep
    # -------------------------------------------------------------------------
    def testDayStep(self):

        pt = ProductTypeMod44(ProductTypeMod44TestCase._inDir)
        self.assertEqual(pt.dayStep, ProductTypeMod44.DAY_STEP)
        
    # -------------------------------------------------------------------------
    # testFindFile
    # -------------------------------------------------------------------------
    def testFindFile(self):
        
        pt = ProductTypeMod44(ProductTypeMod44TestCase._inDir)
        
        tid = 'h09v05'
        year = 2019
        day = 65
        bandName = ProductType.BAND4
        fileName: Path = pt.findFile(tid, year, day, bandName)
        
        expFile = '/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' + \
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf'
                  
        self.assertEqual(str(fileName), expFile)

        # Test a file that should not exist.
        day = 66

        with self.assertRaisesRegex(RuntimeError, 'Unable to find file for'):
            fileName: Path = pt.findFile(tid, year, day, bandName)
        
    # -------------------------------------------------------------------------
    # testInputDir
    # -------------------------------------------------------------------------
    def testInputDir(self):
        
        pt = ProductTypeMod44(ProductTypeMod44TestCase._inDir)
        self.assertEqual(pt.inputDir, ProductTypeMod44TestCase._inDir)

    # -------------------------------------------------------------------------
    # testSolarZenithScaleFactor
    # -------------------------------------------------------------------------
    def testSolarZenithScaleFactor(self):

        pt = ProductTypeMod44(ProductTypeMod44TestCase._inDir)
        self.assertEqual(pt.solarZenithScaleFactor, 1.0)
        
    # -------------------------------------------------------------------------
    # testYearOneDays
    # -------------------------------------------------------------------------
    def testYearOneDays(self):
        
        pt = ProductTypeMod44(ProductTypeMod44TestCase._inDir)

        expDays = [65, 81, 97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 
                   257, 273, 289, 305, 321, 337, 353]
                   
        self.assertEqual(expDays, pt.yearOneDays)
        
    # -------------------------------------------------------------------------
    # testYearTwoDays
    # -------------------------------------------------------------------------
    def testYearTwoDays(self):
        
        pt = ProductTypeMod44(ProductTypeMod44TestCase._inDir)
        expDays = [1, 17, 33, 49]
        self.assertEqual(expDays, pt.yearTwoDays)
        
    # -------------------------------------------------------------------------
    # testYearOneStartDay
    # -------------------------------------------------------------------------
    def testYearOneStartDay(self):

        pt = ProductTypeMod44(ProductTypeMod44TestCase._inDir)
        self.assertEqual(pt.yearOneStartDay, ProductType.YEAR_ONE_START_DAY)

    # -------------------------------------------------------------------------
    # testYearOneEndDay
    # -------------------------------------------------------------------------
    def testYearOneEndDay(self):

        pt = ProductTypeMod44(ProductTypeMod44TestCase._inDir)
        self.assertEqual(pt.yearOneEndDay, ProductTypeMod44.YEAR_ONE_END_DAY)

    # -------------------------------------------------------------------------
    # testYearTwoStartDay
    # -------------------------------------------------------------------------
    def testYearTwoStartDay(self):

        pt = ProductTypeMod44(ProductTypeMod44TestCase._inDir)
        self.assertEqual(pt.yearTwoStartDay, ProductType.YEAR_TWO_START_DAY)

    # -------------------------------------------------------------------------
    # testYearTwoEndDay
    # -------------------------------------------------------------------------
    def testYearTwoEndDay(self):

        pt = ProductTypeMod44(ProductTypeMod44TestCase._inDir)
        self.assertEqual(pt.yearTwoEndDay, ProductTypeMod44.YEAR_TWO_END_DAY)

    # -------------------------------------------------------------------------
    # testBandXref
    # -------------------------------------------------------------------------
    def testBandXref(self):

        pt = ProductTypeMod44(ProductTypeMod44TestCase._inDir)

        chName = pt.inputDir / 'MOD44CH.A2019065.h09v05.061.2020290183523.hdf'
        cqName = pt.inputDir / 'MOD44CQ.A2019065.h09v05.061.2020290183523.hdf'
        
        chDs: gdal.Dataset = gdal.Open(str(chName))
        cqDs: gdal.Dataset = gdal.Open(str(cqName))
        
        chSubs = chDs.GetSubDatasets()
        cqSubs = cqDs.GetSubDatasets()

        # Band 5
        subdatasetIndex: int = pt.bandXref[pt.BAND5]
        desc: str = chSubs[subdatasetIndex][1]
        self.assertTrue('Band_5' in desc)
        
        # SOLZ
        subdatasetIndex: int = pt.bandXref[pt.SOLZ]
        desc: str = cqSubs[subdatasetIndex][1]
        self.assertTrue('solar_zenith' in desc)

        # State
        subdatasetIndex: int = pt.bandXref[pt.STATE]
        desc: str = cqSubs[subdatasetIndex][1]
        self.assertTrue('state' in desc)
        