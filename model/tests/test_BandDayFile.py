
from pathlib import Path
import tempfile
import unittest

import numpy as np

from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.ProductType import ProductType
from modis_vcf.model.ProductTypeMod09A import ProductTypeMod09A
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44


# -----------------------------------------------------------------------------
# class BandDayFileTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_BandDayFile
# python -m unittest modis_vcf.model.tests.test_BandDayFile.BandDayFileTestCase.testInit
# -----------------------------------------------------------------------------
class BandDayFileTestCase(unittest.TestCase):

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
        print(cls._outDir)

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        pt = ProductTypeMod44(BandDayFileTestCase._inDir44)
        tid = 'h09v05'
        year = 2019
        day = 65
        bandName = ProductType.BAND1
        
        bdf = BandDayFile(pt, 
                          tid, 
                          year, 
                          day, 
                          bandName, 
                          BandDayFileTestCase._outDir)
        
        self.assertEqual(bdf._productType.productType, pt.productType)
        self.assertEqual(bdf._tileId, tid)
        self.assertEqual(bdf._year, year)
        self.assertEqual(bdf._day, day)
        self.assertEqual(bdf._bandName, bandName)
        
        outName = BandDayFileTestCase._outDir / \
            '1-Days' / \
            (pt.productType + '-' + str(year) + str(day).zfill(3) + '-' +
             bandName +
             '.bin')
        
        self.assertEqual(bdf._outName, outName)

    # -------------------------------------------------------------------------
    # testMod44Ch (Band 3)
    # -------------------------------------------------------------------------
    def testMod44Ch(self):
        
        pt = ProductTypeMod44(BandDayFileTestCase._inDir44)
        tid = 'h09v05'
        year = 2019
        day = 65
        bandName = ProductType.BAND3
        
        bdf = BandDayFile(pt, 
                          tid, 
                          year, 
                          day, 
                          bandName, 
                          BandDayFileTestCase._outDir)
        
        raster = bdf.getRaster
        self.assertEqual(raster.shape, (4800, 4800))
        self.assertEqual(raster.dtype, np.int16)
        
        # Call it again to test Numpy fromfile.
        raster2 = bdf.getRaster
        self.assertTrue((raster == raster2).all())
        self.assertEqual(raster2.shape, (4800, 4800))
        self.assertEqual(raster2.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testMod09 (Band 1)
    # -------------------------------------------------------------------------
    def testMod09B1(self):
        
        pt = ProductTypeMod09A(BandDayFileTestCase._inDir09,
                               BandDayFileTestCase._inDir44)
        
        tid = 'h09v05'
        year = 2019
        day = 65
        bandName = ProductType.BAND1
        
        bdf = BandDayFile(pt, 
                          tid, 
                          year, 
                          day, 
                          bandName, 
                          BandDayFileTestCase._outDir)
        
        raster = bdf.getRaster
        self.assertEqual(raster.shape, (4800, 4800))
        self.assertEqual(raster.dtype, np.int16)
        
        # Call it again to test Numpy fromfile.
        raster2 = bdf.getRaster
        self.assertTrue((raster == raster2).all())
        self.assertEqual(raster2.shape, (4800, 4800))
        self.assertEqual(raster2.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testMod09 (Band 31)
    # -------------------------------------------------------------------------
    def testMod09B31(self):
        
        pt = ProductTypeMod09A(BandDayFileTestCase._inDir09,
                               BandDayFileTestCase._inDir44)
        
        tid = 'h09v05'
        year = 2019
        day = 65
        bandName = ProductType.BAND31
        
        bdf = BandDayFile(pt, 
                          tid, 
                          year, 
                          day, 
                          bandName, 
                          BandDayFileTestCase._outDir)
        
        raster = bdf.getRaster
        self.assertEqual(raster.shape, (4800, 4800))
        self.assertEqual(raster.dtype, np.int16)

        # Call it again to test Numpy fromfile.
        raster2 = bdf.getRaster
        self.assertTrue((raster == raster2).all())
        self.assertEqual(raster2.shape, (4800, 4800))
        self.assertEqual(raster2.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testToTif
    # -------------------------------------------------------------------------
    def testToTif(self):
        
        pt = ProductTypeMod09A(BandDayFileTestCase._inDir09,
                               BandDayFileTestCase._inDir44)
        
        tid = 'h09v05'
        year = 2019
        day = 65
        bandName = ProductType.BAND5
        
        bdf = BandDayFile(pt, 
                          tid, 
                          year, 
                          day, 
                          bandName, 
                          BandDayFileTestCase._outDir)
        
        bdf.toTif()

    # -------------------------------------------------------------------------
    # testValues
    #
    # (gdalNgmt) [rlgill@ilab203 MOD44C]$ gdallocationinfo -valonly HDF4_EOS:EOS_GRID:"/explore/nobackup/projects/ilab/data/MODIS/MOD44C/MOD44CH.A2019065.h09v05.061.2020290183523.hdf":MOD44C_500m_GRID:Band_5 292 0
    # 6394
    # -------------------------------------------------------------------------
    def testValues(self):
        
        pt = ProductTypeMod44(BandDayFileTestCase._inDir44)
        tid = 'h09v05'
        year = 2019
        day = 65
        bandName = ProductType.BAND5
        
        bdf = BandDayFile(pt, 
                          tid, 
                          year, 
                          day, 
                          bandName, 
                          BandDayFileTestCase._outDir)
        
        x = 0
        y = 292
        
        # Band 5 raw
        raw5, dataType = bdf._readSubdataset(applyNoData=False)
        self.assertEqual(raw5.shape, (4800, 4800))
        self.assertEqual(raw5.dtype, np.int16)
        self.assertEqual(np.min(raw5), -28672)
        self.assertEqual(np.max(raw5), 12812)  # 9979
        numNoData = (raw5 == 36864).sum()  # 0
        self.assertEqual(raw5[x, y], 6394)

        raw5, dataType = bdf._readSubdataset()
        self.assertEqual(raw5.shape, (4800, 4800))
        self.assertEqual(raw5.dtype, np.int16)
        self.assertEqual(np.min(raw5), -28672) 
        self.assertEqual(np.max(raw5), 12812)  # 9979
        self.assertEqual((raw5 == -10001).sum(), numNoData)
        self.assertEqual(raw5[x, y], 6394)
        
        # Solz raw
        solz, dType = bdf._readSubdataset(ProductType.SOLZ, applyNoData=False)
        self.assertEqual(solz.shape, (4800, 4800))
        self.assertEqual(solz.dtype, np.uint8)
        self.assertEqual(np.min(solz), 30) 
        self.assertEqual(np.max(solz), 51)  # No no-data values
        numNoData = (solz == 255).sum()
        self.assertEqual(solz[x, y], 43)
        
        solz, dType = bdf._readSubdataset(ProductType.SOLZ)
        solz = (solz * pt.solarZenithScaleFactor).astype(np.int16)
        self.assertFalse((solz > 72).any())  # No zenith cut offs
        self.assertEqual(solz.shape, (4800, 4800))
        self.assertEqual(solz.dtype, np.int16)
        self.assertEqual(np.min(solz), 30)  # No no-data values 
        self.assertEqual(np.max(solz), 51)
        self.assertEqual((solz == -10001).sum(), numNoData)
        self.assertEqual(solz[x, y], 43)  # Not subject to zenith cut off
        
        # Band 5 proper, without QA
        b5NoQa = bdf._readRaster(applyQa = False)
        self.assertEqual(b5NoQa.shape, (4800, 4800))
        self.assertEqual(b5NoQa.dtype, np.int16)
        self.assertEqual(np.min(b5NoQa), -28672)
        self.assertEqual(np.max(b5NoQa), 12812)  
        self.assertEqual((b5NoQa == -10001).sum(), (raw5 == -10001).sum())
        self.assertEqual(raw5[x, y], 6394)
        
        # State
        state, dType = bdf._readSubdataset(ProductType.STATE, applyNoData=False)
        self.assertEqual(state.shape, (4800, 4800))
        self.assertEqual(state.dtype, np.uint16)
        self.assertEqual(np.min(state), 0) 
        self.assertEqual(np.max(state), 36800)
        numNoData = (state == 65535).sum()
        self.assertEqual(state[x, y], 1)
        
        # QA mask
        qa: np.ndarray = pt.createQaMask(state, solz, 72)
        uniques = np.unique(qa)
        self.assertEqual(len(uniques), 2)
        self.assertTrue(1 in uniques)
        self.assertTrue(-10001 in uniques)
        self.assertEqual(qa[x, y], -10001)  # cloud = 1

        # Band 5 proper, with QA
        bdf._outName.unlink()  # Delete, so non-qa version is not read.
        b5 = bdf._readRaster()
        self.assertEqual(b5.shape, (4800, 4800))
        self.assertEqual(b5.dtype, np.int16)
        self.assertEqual(np.min(b5), -28672)
        self.assertEqual(np.max(b5), 9214)  # unverified 
        self.assertEqual(b5[x, y], -10001)

        # Band 5 proper, with QA
        bdf._outName.unlink()  # Delete, so non-qa version is not read.
        b5 = bdf.getRaster
        self.assertEqual(b5.shape, (4800, 4800))
        self.assertEqual(b5.dtype, np.int16)
        self.assertEqual(np.min(b5), -28672)
        self.assertEqual(np.max(b5), 9214)  # unverified 
        self.assertEqual(b5[x, y], -10001)
        bdf.toTif()

    # -------------------------------------------------------------------------
    # testPrintValues
    # -------------------------------------------------------------------------
    def testPrintValues(self):
        
        pt = ProductTypeMod44(BandDayFileTestCase._inDir44)
        tid = 'h09v05'
        year = 2020
        day = 49
        bandName = ProductType.BAND5
        
        bdf = BandDayFile(pt, 
                          tid, 
                          year, 
                          day, 
                          bandName, 
                          BandDayFileTestCase._outDir)
        
        x = 0
        y = 292
        
        rawBand = bdf._readSubdataset(applyNoData=False)[0]
        print('Raw Band[', x, ',', y, '] w/o no-data =', rawBand[x, y])
        
        rawBand = bdf._readSubdataset()[0]
        print('Raw Band[', x, ',', y, '] =', rawBand[x, y])

        solz = bdf._readSubdataset(ProductType.SOLZ, applyNoData=False)[0]
        print('Solz[', x, ',', y, '] w/o no-data =', solz[x, y])
        
        solz = bdf._readSubdataset(ProductType.SOLZ)[0]
        print('Solz[', x, ',', y, '] =', solz[x, y])

        bandNoQa = bdf._readRaster(applyQa = False)
        print('Band[', x, ',', y, '] w/o QA =', bandNoQa[x, y])

        state = bdf._readSubdataset(ProductType.STATE, applyNoData=False)[0]
        print('State[', x, ',', y, '] =', state[x, y])

        qa: np.ndarray = pt.createQaMask(state, solz, 72)
        print('QA[', x, ',', y, '] =', qa[x, y])

        bdf._outName.unlink()  # Delete, so non-qa version is not read.
        band = bdf.getRaster
        print('Band[', x, ',', y, '] =', band[x, y])
        