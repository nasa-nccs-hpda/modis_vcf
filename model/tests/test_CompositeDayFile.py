from pathlib import Path
import tempfile
import unittest

import numpy as np

from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.CompositeDayFile import CompositeDayFile
from modis_vcf.model.ProductType import ProductType
from modis_vcf.model.ProductTypeMod09A import ProductTypeMod09A
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44


# -----------------------------------------------------------------------------
# class CompositeDayFileTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_CompositeDayFile
# python -m unittest modis_vcf.model.tests.test_CompositeDayFile.CompositeDayFileTestCase.testInit
# -----------------------------------------------------------------------------
class CompositeDayFileTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUp
    # -------------------------------------------------------------------------
    def setUp(self):

        self.h09v05 = 'h09v05'
        self.year2019 = 2019
        day = 65
        bandName = ProductType.BAND1
        
        # MOD44
        self._inDir44 = \
            Path('/css/modis/Collection6.1/L3/MOD44B-VCF/dev')
        
        
        self.productTypeMod44 = ProductTypeMod44(self._inDir44)
        
        self._mod44OutDir = Path('/explore/nobackup/people/rlgill' +      
                                 '/SystemTesting/modis-vcf/MOD44') / \
                            Path(self.h09v05) / \
                            Path(str(self.year2019))

        self._dayDirMod44 = self._mod44OutDir / '1-Days'
        self._compDirMod44 = self._mod44OutDir / '2-Composites'
        self._dayDirMod44.mkdir(exist_ok=True)
        self._compDirMod44.mkdir(exist_ok=True)

        self.cdfMod44 = \
            CompositeDayFile().initFromParams(self.productTypeMod44, 
                                              bandName,
                                              self.h09v05, 
                                              self.year2019, 
                                              day, 
                                              self._compDirMod44,
                                              logger=None,
                                              dayDir=self._dayDirMod44)
                                              
        # MOD09A
        self._inDir09 = \
            Path('/explore/nobackup/projects/ilab/data/MODIS/MOD09A1')
        
        self.productTypeMod09 = ProductTypeMod09A(self._inDir09, self._inDir44)

        self._mod09OutDir = Path('/explore/nobackup/people/rlgill' +      
                                 '/SystemTesting/modis-vcf/MOD09A') / \
                            Path(self.h09v05) / \
                            Path(str(self.year2019))

        self._dayDirMod09 = self._mod09OutDir / '1-Days'
        self._compDirMod09 = self._mod09OutDir / '2-Composites'
        self._dayDirMod09.mkdir(exist_ok=True)
        self._compDirMod09.mkdir(exist_ok=True)

        self.cdfMod09 = \
            CompositeDayFile().initFromParams(self.productTypeMod09, 
                                              bandName,
                                              self.h09v05, 
                                              self.year2019, 
                                              day, 
                                              self._compDirMod09,
                                              logger=None,
                                              dayDir=self._dayDirMod09)

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        CompositeDayFile()
        
    # -------------------------------------------------------------------------
    # testInitFromParams
    # -------------------------------------------------------------------------
    def testInitFromParams(self):
        
        day = 65
        bandName = ProductType.BAND1
        
        cdf = CompositeDayFile().initFromParams(self.productTypeMod44, 
                                                bandName,
                                                self.h09v05, 
                                                self.year2019, 
                                                day, 
                                                self._compDirMod44,
                                                logger=None,
                                                dayDir=self._dayDirMod44)

        self.assertEqual(cdf.productType, self.productTypeMod44)
        self.assertEqual(cdf.tid, self.h09v05)
        self.assertEqual(cdf.year, self.year2019)
        self.assertEqual(cdf.day, day)
        self.assertEqual(cdf.bandName, bandName)
        self.assertEqual(cdf._daysInComp, 32)
        
        outName: Path = self._compDirMod44 / \
                        (self.productTypeMod44.productType +
                         '-' +
                         self.h09v05 +
                         '-' +
                         str(self.year2019) +
                         str(day).zfill(3) +
                         '-' +
                         bandName +
                         '.bin')

        self.assertEqual(cdf.outName, outName)

    # -------------------------------------------------------------------------
    # testCopy
    # -------------------------------------------------------------------------
    def testCopy(self):
        
        day = 66
        cdf = CompositeDayFile().copy(self.cdfMod44, self.year2019, day)
        
        self.assertEqual(cdf.productType.productType,
                         self.productTypeMod44.productType)

        self.assertEqual(cdf.tid, self.h09v05)
        self.assertEqual(cdf.year, self.year2019)
        self.assertEqual(cdf.day, day)
        self.assertEqual(cdf.bandName, self.cdfMod44.bandName)

        outName: Path = self._compDirMod44 / \
                        (self.productTypeMod44.productType +
                         '-' +
                         self.h09v05 +
                         '-' +
                         str(self.year2019) +
                         str(day).zfill(3) +
                         '-' +
                         self.cdfMod44.bandName +
                         '.bin')

        self.assertEqual(cdf.outName, outName)

        cdf = CompositeDayFile().copy(cdf, cdf.year, cdf.day + 1)
        self.assertEqual(cdf.tid, self.h09v05)
        self.assertEqual(cdf.year, self.year2019)
        self.assertEqual(cdf.day, day + 1)
        self.assertEqual(cdf.bandName, self.cdfMod44.bandName)
        
    # -------------------------------------------------------------------------
    # testBandName
    # -------------------------------------------------------------------------
    def testBandName(self):

        self.assertEqual(self.cdfMod44.bandName, ProductType.BAND1)

    # -------------------------------------------------------------------------
    # testDay
    # -------------------------------------------------------------------------
    def testDay(self):

        self.assertEqual(self.cdfMod44.day, 65)

    # -------------------------------------------------------------------------
    # testProductType
    # -------------------------------------------------------------------------
    def testProductType(self):

        self.assertEqual(self.cdfMod44.productType, self.productTypeMod44)

    # -------------------------------------------------------------------------
    # testTid
    # -------------------------------------------------------------------------
    def testTid(self):

        self.assertEqual(self.cdfMod44.tid, self.h09v05)

    # -------------------------------------------------------------------------
    # testYear
    # -------------------------------------------------------------------------
    def testYear(self):

        self.assertEqual(self.cdfMod44.year, self.year2019)

    # -------------------------------------------------------------------------
    # testMod44Ch (Band 3)
    # -------------------------------------------------------------------------
    def testMod44Ch(self):
        
        cdf = CompositeDayFile().initFromParams(self.productTypeMod44, 
                                                ProductType.BAND3,
                                                self.h09v05, 
                                                self.year2019, 
                                                65, 
                                                self._compDirMod44,
                                                logger=None,
                                                dayDir=self._dayDirMod44)
                            
        cdf.outName.unlink(missing_ok=True)                    
        composite = cdf.raster()
        self.assertEqual(composite.shape, (4800, 4800))
        self.assertEqual(composite.dtype, np.int16)
        
        # Call it again to test Numpy fromfile.
        composite2 = cdf.raster()
        self.assertTrue(np.allclose(composite, composite2, equal_nan=True))
        self.assertEqual(composite2.shape, (4800, 4800))
        self.assertEqual(composite2.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testMod09B1 (Band 1)
    # -------------------------------------------------------------------------
    def testMod09B1(self):
        
        self.cdfMod09.outName.unlink(missing_ok=True)                    
        composite = self.cdfMod09.raster()
        self.assertEqual(composite.shape, (4800, 4800))
        self.assertEqual(composite.dtype, np.int16)
        
        # Call it again to test Numpy fromfile.
        composite2 = self.cdfMod09.raster()
        self.assertTrue(np.allclose(composite, composite2, equal_nan=True))
        self.assertEqual(composite2.shape, (4800, 4800))
        self.assertEqual(composite2.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testMod09B31 (Band 31)
    # -------------------------------------------------------------------------
    def testMod09B31(self):

        cdf = CompositeDayFile().initFromParams(self.productTypeMod09, 
                                                ProductType.BAND31,
                                                self.h09v05, 
                                                self.year2019, 
                                                65, 
                                                self._compDirMod09,
                                                logger=None,
                                                dayDir=self._dayDirMod09)
                                                
        cdf.outName.unlink(missing_ok=True)   
        
        # Band 31 is of type uint16.  This will fail.                 
        with self.assertRaisesRegex(ValueError, 'cannot reshape'):
            
            composite = cdf.raster()
            self.assertEqual(composite.dtype, np.int16)
            self.assertEqual(composite.shape, (4800, 4800))
        
            # Call it again to test Numpy fromfile.
            composite2 = cdf.raster()
            self.assertTrue(np.allclose(composite, composite2, equal_nan=True))
            self.assertEqual(composite2.shape, (4800, 4800))
            self.assertEqual(composite2.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testYearWrap
    # -------------------------------------------------------------------------
    def testYearWrap(self):
        
        cdf = CompositeDayFile().initFromParams(self.productTypeMod09, 
                                                ProductType.BAND5,
                                                self.h09v05, 
                                                self.year2019, 
                                                353, 
                                                self._compDirMod09,
                                                logger=None,
                                                dayDir=self._dayDirMod09)

        cdf.outName.unlink(missing_ok=True)                    
        composite = cdf.raster()
        self.assertEqual(composite.shape, (4800, 4800))
        self.assertEqual(composite.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testLastDayOfYear2
    # -------------------------------------------------------------------------
    def testLastDayOfYear2(self):
        
        cdf = CompositeDayFile().initFromParams(self.productTypeMod09, 
                                                ProductType.BAND5,
                                                self.h09v05, 
                                                2020, 
                                                49, 
                                                self._compDirMod09,
                                                logger=None,
                                                dayDir=self._dayDirMod09)

        cdf.outName.unlink(missing_ok=True)                    

        # ---
        # 2020049 is the last day of the 32-day composites starting in 2019.
        # For MOD09 this should be comprised of two day files, 49 and 57.
        # ---
        composite = cdf.raster()
        self.assertEqual(composite.shape, (4800, 4800))
        self.assertEqual(composite.dtype, np.int16)
        
        # Call it again to test Numpy fromfile.
        composite2 = cdf.raster()
        self.assertTrue(np.allclose(composite, composite2, equal_nan=True))
        self.assertEqual(composite2.shape, (4800, 4800))
        self.assertEqual(composite2.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testGetDaysToFindMod09
    # -------------------------------------------------------------------------
    def testGetDaysToFindMod09(self):
        
        cdf = CompositeDayFile().initFromParams(self.productTypeMod09, 
                                                ProductType.BAND5,
                                                self.h09v05, 
                                                self.year2019, 
                                                65, 
                                                self._compDirMod09,
                                                logger=None,
                                                dayDir=self._dayDirMod09)

        expDays = [(2019, 65), (2019, 73), (2019, 81), (2019, 89)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
        # Middle day
        cdf = CompositeDayFile().copy(cdf, self.year2019, 225)
        expDays = [(2019, 225), (2019, 233), (2019, 241), (2019, 249)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
        # End of year 1
        cdf = CompositeDayFile().copy(cdf, self.year2019, 353)
        expDays = [(2019, 353), (2019, 361), (2020, 1), (2020, 9)]
        self.assertEqual(cdf._getDaysToFind(), expDays)

        # Beginning of year 2
        cdf = CompositeDayFile().copy(cdf, 2020, 17)
        expDays = [(2020, 17), (2020, 25), (2020, 33), (2020, 41)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
        # End of year 2
        cdf = CompositeDayFile().copy(cdf, 2020, 49)
        expDays = [(2020, 49), (2020, 57)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
    # -------------------------------------------------------------------------
    # testGetDaysToFindMod44
    # -------------------------------------------------------------------------
    def testGetDaysToFindMod44(self):
        
        # First day
        cdf = CompositeDayFile().initFromParams(self.productTypeMod44, 
                                                ProductType.BAND5,
                                                self.h09v05, 
                                                self.year2019, 
                                                65, 
                                                self._compDirMod44,
                                                logger=None,
                                                dayDir=self._dayDirMod44)

        expDays = [(2019, 65), (2019, 81)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
        # Middle day
        cdf = CompositeDayFile().copy(cdf, 2019, 225)
        expDays = [(2019, 225), (2019, 241)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
        # End of year 1
        cdf = CompositeDayFile().copy(cdf, 2019, 353)
        expDays = [(2019, 353), (2020, 1)]
        self.assertEqual(cdf._getDaysToFind(), expDays)

        # Beginning of year 2
        cdf = CompositeDayFile().copy(cdf, 2020, 17)
        expDays = [(2020, 17), (2020, 33)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
        # End of year 2
        cdf = CompositeDayFile().copy(cdf, 2020, 49)
        expDays = [(2020, 49)]
        self.assertEqual(cdf._getDaysToFind(), expDays)

    # -------------------------------------------------------------------------
    # testCreateComposite
    # -------------------------------------------------------------------------
    def testCreateComposite(self):
        
        day = 65
        bandName = ProductType.BAND5
        
        cdf = CompositeDayFile().initFromParams(self.productTypeMod44, 
                                                bandName,
                                                self.h09v05, 
                                                self.year2019, 
                                                day, 
                                                self._compDirMod44,
                                                logger=None,
                                                dayDir=self._dayDirMod44)

        cdf.outName.unlink(missing_ok=True)                    
        expDays = [(2019, 65), (2019, 81)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        comp = cdf.raster()   
        self.assertEqual(comp.dtype, np.int16)     
        testDays = []
        
        for year, day in expDays:
            
            name = 'MOD44-' + self.h09v05 + '-' + str(self.year2019) + \
                   str(day).zfill(3) + \
                   '-' + bandName + '.bin'
                   
            fName = self._dayDirMod44 / name
            raster = np.fromfile(fName, dtype=np.int16).reshape(4800, 4800)
            testDays.append(raster)
        
        x = 0
        y = 0
        self.assertFalse(np.isnan(testDays[0][x, y]))
        self.assertFalse(np.isnan(testDays[1][x, y]))
        
        self.assertEqual(comp[x, y], \
                         int((testDays[0][x, y] + testDays[1][x, y]) / 2))

        x = 21
        y = 12
        self.assertFalse(np.isnan(testDays[0][x, y]))
        self.assertFalse(np.isnan(testDays[1][x, y]))
        
        self.assertEqual(comp[0, 0], \
                         int((testDays[0][0, 0] + testDays[1][0, 0]) / 2))

        x = 2100
        y = 1200
        self.assertFalse(np.isnan(testDays[0][x, y]))
        self.assertFalse(np.isnan(testDays[1][x, y]))
        
        self.assertEqual(comp[0, 0], \
                         int((testDays[0][0, 0] + testDays[1][0, 0]) / 2))

        x = 4799
        y = 4799
        self.assertFalse(np.isnan(testDays[0][x, y]))
        self.assertFalse(np.isnan(testDays[1][x, y]))
        
        self.assertEqual(comp[0, 0], \
                         int((testDays[0][0, 0] + testDays[1][0, 0]) / 2))
 
    # -------------------------------------------------------------------------
    # testSolz
    #
    # The metrics in h12v02 suggested an erroneous composition of day files
    # related to solar zenith.
    # -------------------------------------------------------------------------
    def testSolz(self):
 
        cdf = CompositeDayFile().initFromParams(self.productTypeMod09, 
                                                ProductType.BAND1,
                                                'h12v02', 
                                                self.year2019, 
                                                289, 
                                                self._compDirMod44,
                                                logger=None,
                                                dayDir=self._dayDirMod44)

        cdf.raster()

    # -------------------------------------------------------------------------
    # testRead
    # -------------------------------------------------------------------------
    def testRead(self):
 
        cdf = CompositeDayFile().initFromParams(self.productTypeMod09, 
                                                ProductType.BAND1,
                                                'h12v02', 
                                                self.year2019, 
                                                289, 
                                                self._compDirMod44,
                                                logger=None,
                                                dayDir=self._dayDirMod44)

        r1 = cdf.raster()
        self.assertEqual(r1.shape, (4800, 4800))
        self.assertEqual(r1.dtype, np.int16)
        r2 = cdf.raster()
        self.assertTrue(np.array_equal(r1, r2, equal_nan=True))
        cdf._raster = None
        r3 = cdf.raster()
        self.assertTrue(np.array_equal(r1, r3, equal_nan=True))
        