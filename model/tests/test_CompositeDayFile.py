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
    # setUpClass
    # -------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):

        cls._inDir44 = \
            Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')
        
        cls._inDir09 = \
            Path('/explore/nobackup/projects/ilab/data/MODIS/MOD09A1')
        
        cls._outDir = Path(tempfile.mkdtemp())
        cls._dayDir = cls._outDir / '1-Days'
        cls._compDir = cls._outDir / '2-Composites'
        cls._dayDir.mkdir()
        cls._compDir.mkdir()
        
        print(cls._outDir)
        
    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        pt = ProductTypeMod44(CompositeDayFileTestCase._inDir44)
        tid = 'h09v05'
        year = 2019
        day = 65
        bandName = ProductType.BAND1
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        self.assertEqual(cdf._productType.productType, pt.productType)
        self.assertEqual(cdf._tileId, tid)
        self.assertEqual(cdf._year, year)
        self.assertEqual(cdf._day, day)
        self.assertEqual(cdf._bandName, bandName)
        self.assertEqual(cdf._daysInComp, 32)
        
        outName: Path = CompositeDayFileTestCase._compDir / \
                        (pt.productType +
                         '-' +
                         tid +
                         '-' +
                         str(year) +
                         str(day).zfill(3) +
                         '-' +
                         bandName +
                         '.bin')

        self.assertEqual(cdf._outName, outName)

    # -------------------------------------------------------------------------
    # testMod44Ch (Band 3)
    # -------------------------------------------------------------------------
    def testMod44Ch(self):
        
        pt = ProductTypeMod44(CompositeDayFileTestCase._inDir44)
        tid = 'h09v05'
        year = 2019
        day = 65
        bandName = ProductType.BAND3
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)

        composite = cdf.getRaster
        self.assertEqual(composite.shape, (4800, 4800))
        self.assertEqual(composite.dtype, np.int16)
        
        # Call it again to test Numpy fromfile.
        composite2 = cdf.getRaster
        self.assertTrue(np.allclose(composite, composite2, equal_nan=True))
        self.assertEqual(composite2.shape, (4800, 4800))
        self.assertEqual(composite2.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testMod09B1 (Band 1)
    # -------------------------------------------------------------------------
    def testMod09B1(self):
        
        pt = ProductTypeMod09A(CompositeDayFileTestCase._inDir09,
                               CompositeDayFileTestCase._inDir44)

        tid = 'h09v05'
        year = 2019
        day = 65
        bandName = ProductType.BAND1
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        composite = cdf.getRaster
        self.assertEqual(composite.shape, (4800, 4800))
        self.assertEqual(composite.dtype, np.int16)
        
        # Call it again to test Numpy fromfile.
        composite2 = cdf.getRaster
        self.assertTrue(np.allclose(composite, composite2, equal_nan=True))
        self.assertEqual(composite2.shape, (4800, 4800))
        self.assertEqual(composite2.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testMod09B31 (Band 31)
    # -------------------------------------------------------------------------
    def testMod09B31(self):
        
        pt = ProductTypeMod09A(CompositeDayFileTestCase._inDir09,
                               CompositeDayFileTestCase._inDir44)
        
        tid = 'h09v05'
        year = 2019
        day = 65
        bandName = ProductType.BAND31
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        composite = cdf.getRaster
        self.assertEqual(composite.shape, (4800, 4800))
        self.assertEqual(composite.dtype, np.int16)
        
        # Call it again to test Numpy fromfile.
        composite2 = cdf.getRaster
        self.assertTrue(np.allclose(composite, composite2, equal_nan=True))
        self.assertEqual(composite2.shape, (4800, 4800))
        self.assertEqual(composite2.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testYearWrap
    # -------------------------------------------------------------------------
    def testYearWrap(self):
        
        pt = ProductTypeMod09A(CompositeDayFileTestCase._inDir09,
                               CompositeDayFileTestCase._inDir44)
        
        tid = 'h09v05'
        year = 2019
        day = 353
        bandName = ProductType.BAND5
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        composite = cdf.getRaster
        self.assertEqual(composite.shape, (4800, 4800))
        self.assertEqual(composite.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testLastDayOfYear2
    # -------------------------------------------------------------------------
    def testLastDayOfYear2(self):
        
        pt = ProductTypeMod09A(CompositeDayFileTestCase._inDir09,
                               CompositeDayFileTestCase._inDir44)
        
        tid = 'h09v05'
        year = 2020
        day = 49
        bandName = ProductType.BAND5
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        # ---
        # 2020049 is the last day of the 32-day composites starting in 2019.
        # For MOD09 this should be comprised of two day files, 49 and 57.
        # ---
        composite = cdf.getRaster
        self.assertEqual(composite.shape, (4800, 4800))
        self.assertEqual(composite.dtype, np.int16)
        
        # Call it again to test Numpy fromfile.
        composite2 = cdf.getRaster
        self.assertTrue(np.allclose(composite, composite2, equal_nan=True))
        self.assertEqual(composite2.shape, (4800, 4800))
        self.assertEqual(composite2.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testGetDaysToFindMod09
    # -------------------------------------------------------------------------
    def testGetDaysToFindMod09(self):
        
        pt = ProductTypeMod09A(CompositeDayFileTestCase._inDir09,
                               CompositeDayFileTestCase._inDir44)
        
        tid = 'h09v05'
        bandName = ProductType.BAND5

        # First day
        year = 2019
        day = 65
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        expDays = [(2019, 65), (2019, 73), (2019, 81), (2019, 89)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
        # Middle day
        year = 2019
        day = 225
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        expDays = [(2019, 225), (2019, 233), (2019, 241), (2019, 249)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
        # End of year 1
        year = 2019
        day = 353
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        expDays = [(2019, 353), (2019, 361), (2020, 1), (2020, 9)]
        self.assertEqual(cdf._getDaysToFind(), expDays)

        # Beginning of year 2
        year = 2020
        day = 17
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        expDays = [(2020, 17), (2020, 25), (2020, 33), (2020, 41)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
        # End of year 2
        year = 2020
        day = 49
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        expDays = [(2020, 49), (2020, 57)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
    # -------------------------------------------------------------------------
    # testGetDaysToFindMod44
    # -------------------------------------------------------------------------
    def testGetDaysToFindMod44(self):
        
        pt = ProductTypeMod44(CompositeDayFileTestCase._inDir44)
        tid = 'h09v05'
        bandName = ProductType.BAND5

        # First day
        year = 2019
        day = 65
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        expDays = [(2019, 65), (2019, 81)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
        # Middle day
        year = 2019
        day = 225
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        expDays = [(2019, 225), (2019, 241)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
        # End of year 1
        year = 2019
        day = 353
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        expDays = [(2019, 353), (2020, 1)]
        self.assertEqual(cdf._getDaysToFind(), expDays)

        # Beginning of year 2
        year = 2020
        day = 17
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        expDays = [(2020, 17), (2020, 33)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        
        # End of year 2
        year = 2020
        day = 49
        
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        expDays = [(2020, 49)]
        self.assertEqual(cdf._getDaysToFind(), expDays)

    # -------------------------------------------------------------------------
    # testCreateComposite
    # -------------------------------------------------------------------------
    def testCreateComposite(self):
        
        # Get the expected bands.
        pt = ProductTypeMod44(CompositeDayFileTestCase._inDir44)
        tid = 'h09v05'
        year = 2019
        day = 65
        bandName = ProductType.BAND5
        
        # Create the composite.
        cdf = CompositeDayFile(pt, 
                               tid, 
                               year, 
                               day, 
                               bandName,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)
        
        expDays = [(2019, 65), (2019, 81)]
        self.assertEqual(cdf._getDaysToFind(), expDays)
        comp = cdf.getRaster   
        self.assertEqual(comp.dtype, np.int16)     
        testDays = []
        
        for year, day in expDays:
            
            name = 'MOD44-' + tid + '-' + str(year) + str(day).zfill(3) + \
                   '-' + bandName + '.bin'
                   
            fName = CompositeDayFileTestCase._dayDir / name
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
 
        inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD09A1')

        outDir = Path('/explore/nobackup/people/rlgill' +      
                      '/SystemTesting/modis-vcf/MOD09A')

        pt = ProductTypeMod09A(inDir, CompositeDayFileTestCase._inDir44)

        cdf = CompositeDayFile(pt, 
                               'h12v02', 
                               2019, 
                               289, 
                               pt.BAND1,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)

        cdf.getRaster

    # -------------------------------------------------------------------------
    # testRead
    # -------------------------------------------------------------------------
    def testRead(self):
 
        inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD09A1')
        pt = ProductTypeMod09A(inDir, CompositeDayFileTestCase._inDir44)

        cdf = CompositeDayFile(pt, 
                               'h12v02', 
                               2019, 
                               289, 
                               pt.BAND1,
                               CompositeDayFileTestCase._compDir,
                               CompositeDayFileTestCase._dayDir)

        r1 = cdf.getRaster
        self.assertEqual(r1.shape, (4800, 4800))
        self.assertEqual(r1.dtype, np.int16)
        r2 = cdf.getRaster
        self.assertTrue(np.array_equal(r1, r2, equal_nan=True))
        cdf._raster = None
        r3 = cdf.getRaster
        self.assertTrue(np.array_equal(r1, r3, equal_nan=True))
        