
import logging
from pathlib import Path
import sys
import shutil
import tempfile
import unittest

import pytest

import numpy as np

from modis_vcf.model import Settings
from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.ProductType import ProductType
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44


# -----------------------------------------------------------------------------
# class BandDayFileTestCase
#
# pytest tests/test_BandDayFile.py
# pytest tests/test_BandDayFile.py::BandDayFileTestCase::testInit
#
# TODO: Test solar zenith for 16000s.
# -----------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.skipif(Settings.MOD44_COLLECTION != '61', reason='reads Collection 6.1 .hdf inputs: set VCF_MOD44_COLLECTION=61')
class BandDayFileTestCase(unittest.TestCase):

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

        self._mod44OutDir = Path(tempfile.mkdtemp(prefix='vcf_test_'))  # removed after the test
        self.addCleanup(shutil.rmtree, str(self._mod44OutDir), True)
        self._mod44OutDir.mkdir(exist_ok=True)
        
        self.productTypeMod44 = ProductTypeMod44(self._mod44InDir)

        day = 65
        bandName = ProductType.BAND1

        self.bdfMod44 = BandDayFile().initFromParams(self.productTypeMod44,
                                                     bandName,
                                                     self.h09v05,
                                                     self.year2019,
                                                     day,
                                                     self._mod44OutDir)

        self.days = [(2019,  65), (2019,  97), (2019, 129), (2019, 161),
                    (2019, 193), (2019, 225), (2019, 257), (2019, 289),
                    (2019, 321), (2019, 353), (2020,  17), (2020, 49)]

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        bdf = BandDayFile()

    # -------------------------------------------------------------------------
    # testInitFromParams
    # -------------------------------------------------------------------------
    def testInitFromParams(self):
        
        day = 65
        bandName = ProductType.BAND1

        bdf = BandDayFile().initFromParams(self.productTypeMod44,
                                           bandName,
                                           self.h09v05,
                                           self.year2019,
                                           day,
                                           self._mod44OutDir)

        self.assertEqual(bdf.productType.productType,
                         self.productTypeMod44.productType)

        self.assertEqual(bdf.tid, self.h09v05)
        self.assertEqual(bdf.year, self.year2019)
        self.assertEqual(bdf.day, day)
        self.assertEqual(bdf.bandName, bandName)
        
        expDir = self._mod44OutDir
        
        self.assertEqual(bdf._outDir, expDir)

        outName: Path = self._mod44OutDir / \
                        (self.productTypeMod44.productType +
                         '-' +
                         self.h09v05 +
                         '-' +
                         str(self.year2019) +
                         str(day).zfill(3) +
                         '-' +
                         bandName +
                         '.bin')

        self.assertEqual(bdf.outName, outName)

    # -------------------------------------------------------------------------
    # testCopy
    # -------------------------------------------------------------------------
    def testCopy(self):
        
        bandName = ProductType.BAND1
        day = 66
        bdf = BandDayFile().copy(self.bdfMod44, self.year2019, day)
        
        self.assertEqual(bdf.productType.productType,
                         self.productTypeMod44.productType)

        self.assertEqual(bdf.tid, self.h09v05)
        self.assertEqual(bdf.year, self.year2019)
        self.assertEqual(bdf.day, day)
        self.assertEqual(bdf.bandName, bandName)

        outName: Path = self._mod44OutDir / \
                        (self.productTypeMod44.productType +
                         '-' +
                         self.h09v05 +
                         '-' +
                         str(self.year2019) +
                         str(day).zfill(3) +
                         '-' +
                         bandName +
                         '.bin')

        self.assertEqual(bdf.outName, outName)

    # -------------------------------------------------------------------------
    # testBandName
    # -------------------------------------------------------------------------
    def testBandName(self):

        self.assertEqual(self.bdfMod44.bandName, ProductType.BAND1)

    # -------------------------------------------------------------------------
    # testDay
    # -------------------------------------------------------------------------
    def testDay(self):

        self.assertEqual(self.bdfMod44.day, 65)

    # -------------------------------------------------------------------------
    # testOutName
    # -------------------------------------------------------------------------
    def testOutName(self):
        
        expName = self._mod44OutDir / \
                  (self.bdfMod44.productType.productType + 
                   '-' +
                   self.h09v05 +
                   '-' +
                   str(self.year2019) +
                   str(65).zfill(3) +
                   '-' +
                   ProductType.BAND1 +
                   '.bin')

        self.assertEqual(self.bdfMod44.outName, expName)
        
    # -------------------------------------------------------------------------
    # testProductType
    # -------------------------------------------------------------------------
    def testProductType(self):

        self.assertEqual(self.bdfMod44.productType, self.productTypeMod44)

    # -------------------------------------------------------------------------
    # testTid
    # -------------------------------------------------------------------------
    def testTid(self):

        self.assertEqual(self.bdfMod44.tid, self.h09v05)

    # -------------------------------------------------------------------------
    # testYear
    # -------------------------------------------------------------------------
    def testYear(self):

        self.assertEqual(self.bdfMod44.year, self.year2019)

    # -------------------------------------------------------------------------
    # testMod44Ch (Band 3)
    # -------------------------------------------------------------------------
    def testMod44Ch(self):
        
        day = 65
        bandName = ProductType.BAND3
        
        bdf = BandDayFile().initFromParams(self.productTypeMod44,
                                           bandName,
                                           self.h09v05,
                                           self.year2019,
                                           day,
                                           self._mod44OutDir)

        raster = bdf.raster()
        self.assertEqual(raster.shape, (4800, 4800))
        self.assertEqual(raster.dtype, np.int16)
        
        # Call it again to test Numpy fromfile.
        raster2 = bdf.raster()
        self.assertTrue((raster == raster2).all())
        self.assertEqual(raster2.shape, (4800, 4800))
        self.assertEqual(raster2.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testB31
    # -------------------------------------------------------------------------
    def testB31(self):
        
        day = 65
        bandName = ProductType.BAND31
        
        bdf = BandDayFile().initFromParams(self.productTypeMod44,
                                           bandName,
                                           'h20v06',
                                           self.year2019,
                                           day,
                                           self._mod44OutDir)

        bdf.outName.unlink(missing_ok=True)
        raster = bdf.raster()
        self.assertEqual(raster.shape, (4800, 4800))
        self.assertEqual(raster.dtype, np.int16)
        self.assertEqual(raster.max(), 31883)

        # Call it again to test Numpy fromfile.
        raster2 = bdf.raster()
        self.assertTrue((raster == raster2).all())
        self.assertEqual(raster2.shape, (4800, 4800))
        self.assertEqual(raster2.dtype, np.int16)
        
    # -------------------------------------------------------------------------
    # testValues
    #
    # (gdalNgmt) [rlgill@ilab203 MOD44C]$ gdallocationinfo -valonly HDF4_EOS:EOS_GRID:"/explore/nobackup/projects/ilab/data/MODIS/MOD44C/MOD44CH.A2019065.h09v05.061.2020290183523.hdf":MOD44C_500m_GRID:Band_5 292 0
    # 6394
    # -------------------------------------------------------------------------
    def testValues(self):
        
        day = 65
        bandName = ProductType.BAND5
        
        bdf = BandDayFile().initFromParams(self.productTypeMod44,
                                           bandName,
                                           self.h09v05,
                                           self.year2019,
                                           day,
                                           self._mod44OutDir)

        x = 0
        y = 292
        
        # Band 5 raw
        raw5 = bdf._readSubdataset()
        self.assertEqual(raw5.shape, (4800, 4800))
        self.assertEqual(raw5.dtype, np.int16)
        self.assertEqual(np.min(raw5), -28672)
        self.assertEqual(np.max(raw5), 12812)
        numNoData = (raw5 == 36864).sum()  # 0
        self.assertEqual(raw5[x, y], 6394)

        raw5 = bdf._readSubdataset()
        self.assertEqual(raw5.shape, (4800, 4800))
        self.assertEqual(raw5.dtype, np.int16)
        self.assertEqual(np.min(raw5), -28672) 
        self.assertEqual(np.max(raw5), 12812)
        self.assertEqual(raw5[x, y], 6394)
        
        # Solz raw
        solz = bdf._readSubdataset(ProductType.SOLZ)
        self.assertEqual(solz.shape, (4800, 4800))
        self.assertEqual(solz.dtype, np.uint8)
        self.assertEqual(np.min(solz), 30) 
        self.assertEqual(np.max(solz), 51)  # No no-data values
        numNoData = (solz == 255).sum()
        self.assertEqual(solz[x, y], 43)
        
        solz = bdf._readSubdataset(ProductType.SOLZ)
        
        solz = (solz * \
                self.productTypeMod44.solarZenithScaleFactor).astype(np.int16)
        
        self.assertFalse((solz > 72).any())  # No zenith cut offs
        self.assertEqual(solz.shape, (4800, 4800))
        self.assertEqual(solz.dtype, np.int16)
        self.assertEqual(np.min(solz), 30)  # No no-data values 
        self.assertEqual(np.max(solz), 51)
        self.assertEqual((solz == -10001).sum(), numNoData)
        self.assertEqual(solz[x, y], 43)  # Not subject to zenith cut off
        
        # Band 5 proper, without QA
        b5NoQa = bdf._getRaster(applyQa = False)
        self.assertEqual(b5NoQa.shape, (4800, 4800))
        self.assertEqual(b5NoQa.dtype, np.int16)
        self.assertEqual(np.min(b5NoQa), -28672)
        self.assertEqual(np.max(b5NoQa), 12812)  
        self.assertEqual(raw5[x, y], 6394)
        
        # State
        state = bdf._readSubdataset(ProductType.STATE)
        self.assertEqual(state.shape, (4800, 4800))
        self.assertEqual(state.dtype, np.uint16)
        self.assertEqual(np.min(state), 0) 
        self.assertEqual(np.max(state), 36800)
        numNoData = (state == 65535).sum()
        self.assertEqual(state[x, y], 1)
        