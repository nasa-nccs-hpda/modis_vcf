
from pathlib import Path
import unittest

import numpy as np

from osgeo import gdal

from modis_vcf.model.ProductType import ProductType
from modis_vcf.model.ProductTypeMod09A import ProductTypeMod09A
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44


# -----------------------------------------------------------------------------
# class ProductTypeMod09ATestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_ProductTypeMod09A
# python -m unittest modis_vcf.model.tests.test_ProductTypeMod09A.ProductTypeMod09ATestCase.testInit
# -----------------------------------------------------------------------------
class ProductTypeMod09ATestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUpClass
    # -------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):

        cls._inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD09A1')
        cls._auxDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)
                               
        self.assertEqual(pt.inputDir, ProductTypeMod09ATestCase._inDir)
        self.assertEqual(pt.productType, 'MOD09')

    # -------------------------------------------------------------------------
    # testCreateQaMask
    # -------------------------------------------------------------------------
    def testCreateQaMask(self):
        
        inFile = ProductTypeMod09ATestCase._inDir / \
                 'MOD09A1.A2019065.h09v05.061.2020289140850.hdf'

        ds: gdal.Dataset = gdal.Open(str(inFile))

        # Get the state, which does not use the no-data value.
        stateIndex = 11
        bandDs = gdal.Open(ds.GetSubDatasets()[stateIndex][0])
        
        state: np.ndarray = bandDs.ReadAsArray(buf_xsize=ProductType.COLS,
                                               buf_ysize=ProductType.ROWS)
                                               
        self.assertEqual(state.shape, (4800, 4800))
        self.assertEqual(state.dtype, np.uint16)
        self.assertEqual(np.min(state), 8) 
        self.assertEqual(np.max(state), 45084)

        # Get the solar zenith, which uses the no-data value.
        solzIndex = 8
        bandDs = gdal.Open(ds.GetSubDatasets()[solzIndex][0])
        bandNoData = bandDs.GetRasterBand(1).GetNoDataValue()
        
        solz: np.ndarray = bandDs.ReadAsArray(buf_xsize=ProductType.COLS,
                                              buf_ysize=ProductType.ROWS)
                                              
        # solz = np.where(solz == bandNoData, -10001, solz)

        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)

        solz = (solz * pt.solarZenithScaleFactor).astype(np.int16)

        self.assertEqual(solz.shape, (4800, 4800))
        self.assertEqual(solz.dtype, np.int16)
        self.assertEqual(np.min(solz), 35) 
        self.assertEqual(np.max(solz), 52)

        # Aerosols are high.
        cloudMixed = state & ProductTypeMod09A.CLOUD_MIXED
        self.assertFalse((cloudMixed == ProductTypeMod09A.CLOUD_MIXED).all())
        cloudy = state & ProductTypeMod09A.CLOUDY
        self.assertFalse((cloudy == ProductTypeMod09A.CLOUDY).all())
        shadow = state & ProductTypeMod09A.CLOUD_SHADOW
        self.assertFalse((shadow == ProductTypeMod09A.CLOUD_SHADOW).all())
        cloudInternal = state & ProductTypeMod09A.CLOUD_INT
        self.assertFalse((cloudInternal == ProductTypeMod09A.CLOUD_INT).all())
        aerosol = state & ProductTypeMod09A.AERO_MASK
        self.assertFalse((aerosol == ProductTypeMod09A.AERO_HIGH).all())
        
        solzCutOff = solz > 72
        self.assertFalse((solzCutOff == True).all())
        
        # Create the mask.
        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)

        qaMask: np.ndarray = pt.createQaMask(state, solz, 72)
        self.assertFalse((qaMask == -10001).all())
        
    # -------------------------------------------------------------------------
    # testDayStep
    # -------------------------------------------------------------------------
    def testDayStep(self):

        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)

        self.assertEqual(pt.dayStep, ProductTypeMod09A.DAY_STEP)
        
    # -------------------------------------------------------------------------
    # testFindFile
    # -------------------------------------------------------------------------
    def testFindFile(self):
        
        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)

        tid = 'h09v05'
        year = 2019
        day = 65
        bandName = ProductType.BAND31
        fileName: Path = pt.findFile(tid, year, day, bandName)
        
        expFile = '/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' + \
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf'
        
        self.assertEqual(str(fileName), expFile)
        
    # -------------------------------------------------------------------------
    # testGetProductTypeForBand
    # -------------------------------------------------------------------------
    def testGetProductTypeForBand(self):
        
        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)

        pt1 = pt.getProductTypeForBand(ProductType.BAND1)
        self.assertEqual(pt1.productType, pt.productType)
        
        pt31 = pt.getProductTypeForBand(ProductType.BAND31)
        expPt31 = ProductTypeMod44(ProductTypeMod09ATestCase._auxDir)
        self.assertEqual(pt31.productType, expPt31.productType)
        
    # -------------------------------------------------------------------------
    # testInputDir
    # -------------------------------------------------------------------------
    def testInputDir(self):
        
        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)

        self.assertEqual(pt.inputDir, ProductTypeMod09ATestCase._inDir)

    # -------------------------------------------------------------------------
    # testSolarZenithScaleFactor
    # -------------------------------------------------------------------------
    def testSolarZenithScaleFactor(self):

        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)
        
        self.assertEqual(pt.solarZenithScaleFactor,
                         ProductTypeMod09A.SOLAR_ZENITH_SCALE_FACTOR)
        
    # -------------------------------------------------------------------------
    # testYearOneDays
    # -------------------------------------------------------------------------
    def testYearOneDays(self):
        
        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)

        expDays = [65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161,
                   169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249, 257,
                   265, 273, 281, 289, 297, 305, 313, 321, 329, 337, 345, 353,
                   361]
                   
        self.assertEqual(expDays, pt.yearOneDays)
        
    # -------------------------------------------------------------------------
    # testYearTwoDays
    # -------------------------------------------------------------------------
    def testYearTwoDays(self):
        
        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)

        expDays = [1, 9, 17, 25, 33, 41, 49, 57]
        self.assertEqual(expDays, pt.yearTwoDays)
        
    # -------------------------------------------------------------------------
    # testYearOneStartDay
    # -------------------------------------------------------------------------
    def testYearOneStartDay(self):

        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)

        self.assertEqual(pt.yearOneStartDay, ProductType.YEAR_ONE_START_DAY)

    # -------------------------------------------------------------------------
    # testYearOneEndDay
    # -------------------------------------------------------------------------
    def testYearOneEndDay(self):

        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)

        self.assertEqual(pt.yearOneEndDay, ProductTypeMod09A.YEAR_ONE_END_DAY)

    # -------------------------------------------------------------------------
    # testYearTwoStartDay
    # -------------------------------------------------------------------------
    def testYearTwoStartDay(self):

        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)

        self.assertEqual(pt.yearTwoStartDay, ProductType.YEAR_TWO_START_DAY)

    # -------------------------------------------------------------------------
    # testYearTwoEndDay
    # -------------------------------------------------------------------------
    def testYearTwoEndDay(self):

        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)

        self.assertEqual(pt.yearTwoEndDay, ProductTypeMod09A.YEAR_TWO_END_DAY)

    # -------------------------------------------------------------------------
    # testBandXref
    # -------------------------------------------------------------------------
    def testBandXref(self):

        pt = ProductTypeMod09A(ProductTypeMod09ATestCase._inDir,
                               ProductTypeMod09ATestCase._auxDir)

        a1Name = pt.inputDir / 'MOD09A1.A2019065.h09v05.061.2020289140850.hdf'
        a1Ds: gdal.Dataset = gdal.Open(str(a1Name))
        a1Subs = a1Ds.GetSubDatasets()

        # Band 5
        subdatasetIndex: int = pt.bandXref[pt.BAND5]
        desc: str = a1Subs[subdatasetIndex][1]
        self.assertTrue('sur_refl_b05' in desc)
        
        # SOLZ
        subdatasetIndex: int = pt.bandXref[pt.SOLZ]
        desc: str = a1Subs[subdatasetIndex][1]
        self.assertTrue('sur_refl_szen' in desc)
                        
        # State
        subdatasetIndex: int = pt.bandXref[pt.STATE]
        desc: str = a1Subs[subdatasetIndex][1]
        self.assertTrue('sur_refl_state_500m' in desc)
        