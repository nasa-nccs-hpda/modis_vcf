
import logging
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from osgeo import gdal

from modis_vcf.model.Band import Band
from modis_vcf.model.Mate import Mate
from modis_vcf.model.Pair import Pair


# -----------------------------------------------------------------------------
# class PairTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_Pair
# python -m unittest modis_vcf.model.tests.test_Pair.PairTestCase.testImageValidity
# -----------------------------------------------------------------------------
class PairTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUpClass
    # -------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):

        cls.ch = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                      'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')

        cls.chMate = Mate(PairTestCase.ch)

        cls.cq = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                      'MOD44CQ.A2019065.h09v05.061.2020290183523.hdf')

        cls.cqMate = Mate(PairTestCase.cq)        
        cls.pair = Pair(PairTestCase.chMate, PairTestCase.cqMate)
        
    # -------------------------------------------------------------------------
    # testQaMask
    # -------------------------------------------------------------------------
    def testQaMask(self):

        qaMask = PairTestCase.pair.qaMask
        
        # Test bug.
        qaMask2 = qaMask

    # -------------------------------------------------------------------------
    # testWriteMask
    # -------------------------------------------------------------------------
    def testWriteMask(self):

        outDir = Path(tempfile.mkdtemp())
        PairTestCase.pair.writeMask(outDir)

    # -------------------------------------------------------------------------
    # testGetMate
    # -------------------------------------------------------------------------
    def testGetMate(self):
        
        mate, index = PairTestCase.pair._getMate(Pair.BAND1)
        self.assertEqual(mate.fileName.name, PairTestCase.cqMate.fileName.name)
        self.assertEqual(index, 4)

    # -------------------------------------------------------------------------
    # testRead
    # -------------------------------------------------------------------------
    def testRead(self):
        
        band, dataType = PairTestCase.pair.read(Pair.BAND1)
        self.assertEqual(band.shape, (4800, 4800))
        self.assertEqual(dataType, gdal.GDT_Int16)
        self.assertIsInstance(band, np.ndarray)
        self.assertEqual(band.dtype, np.int16)

        numNoData = (band == Band.NO_DATA).any()
        print('Are there no-data values in band 1? ' + str(numNoData))
        
        band, dataType = PairTestCase.pair.read(Pair.BAND1, applyQa=True)
        self.assertEqual(band.shape, (4800, 4800))
        self.assertEqual(dataType, gdal.GDT_Int16)

    # -------------------------------------------------------------------------
    # testImageValidity
    # -------------------------------------------------------------------------
    def testImageValidity(self):
        
        # Write a band with QA.
        bandName = Pair.BAND1
        outDir = Path(tempfile.mkdtemp())
        raster, dataType = PairTestCase.pair.read(bandName, applyQa=True)
        cube = np.full((1, 4800, 4800), np.nan)
        cube[0] = raster
        band = Band(bandName, cube, {'2019353': 0})
        bandPath = band.write(outDir)
        
        # Write a band without QA.
        raster, dataType = PairTestCase.pair.read(bandName)
        cube = np.full((1, 4800, 4800), np.nan)
        cube[0] = raster
        band = Band(bandName + '-no-qa', cube, {'2019353': 0})
        bandPath = band.write(outDir)
        
        # Write the QA 
        PairTestCase.pair.writeMask(outDir)