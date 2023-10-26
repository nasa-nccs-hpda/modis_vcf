
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
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        ch = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')

        chMate = Mate(ch)

        cq = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CQ.A2019065.h09v05.061.2020290183523.hdf')

        cqMate = Mate(cq)        
        pair = Pair(chMate, cqMate)

    # -------------------------------------------------------------------------
    # testQaMask
    # -------------------------------------------------------------------------
    def testQaMask(self):

        ch = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')

        chMate = Mate(ch)

        cq = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CQ.A2019065.h09v05.061.2020290183523.hdf')

        cqMate = Mate(cq)        
        pair = Pair(chMate, cqMate)
        qaMask = pair.qaMask
        
        # Test bug.
        qaMask2 = pair.qaMask

    # -------------------------------------------------------------------------
    # testWriteMask
    # -------------------------------------------------------------------------
    def testWriteMask(self):

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        ch = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')

        chMate = Mate(ch)

        cq = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CQ.A2019065.h09v05.061.2020290183523.hdf')

        cqMate = Mate(cq)        
        pair = Pair(chMate, cqMate, logger)
        outDir = Path(tempfile.mkdtemp())
        pair.writeMask(outDir)

    # -------------------------------------------------------------------------
    # testGetMate
    # -------------------------------------------------------------------------
    def testGetMate(self):
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        ch = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')

        chMate = Mate(ch)

        cq = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CQ.A2019065.h09v05.061.2020290183523.hdf')

        cqMate = Mate(cq)        
        pair = Pair(chMate, cqMate, logger)

        mate, index = pair._getMate(Pair.BAND1)
        self.assertEqual(mate.fileName.name, cqMate.fileName.name)
        self.assertEqual(index, 5)

    # -------------------------------------------------------------------------
    # testRead
    # -------------------------------------------------------------------------
    def testRead(self):
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        ch = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')

        chMate = Mate(ch)

        cq = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CQ.A2019065.h09v05.061.2020290183523.hdf')

        cqMate = Mate(cq)        
        pair = Pair(chMate, cqMate, logger)

        band, dataType = pair.read(Pair.BAND1)
        self.assertEqual(band.shape, (4800, 4800))
        self.assertEqual(dataType, gdal.GDT_Int16)
        
        band, dataType = pair.read(Pair.BAND1, applyQa=True)

    # -------------------------------------------------------------------------
    # testImageValidity
    # -------------------------------------------------------------------------
    def testImageValidity(self):
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        ch = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CH.A2019353.h08v04.061.2020323092054.hdf')

        cq = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CQ.A2019353.h08v04.061.2020323092054.hdf')

        chMate = Mate(ch)
        cqMate = Mate(cq)        
        pair = Pair(chMate, cqMate, logger)

        # Write a band with QA.
        bandName = Pair.BAND1
        outDir = Path(tempfile.mkdtemp())
        raster, dataType = pair.read(bandName, applyQa=True)
        cube = np.full((1, 4800, 4800), np.nan)
        cube[0] = raster
        band = Band(bandName, cube, {'2019353': 0})
        bandPath = band.write(outDir)
        
        # Write a band without QA.
        raster, dataType = pair.read(bandName)
        cube = np.full((1, 4800, 4800), np.nan)
        cube[0] = raster
        band = Band(bandName + '-no-qa', cube, {'2019353': 0})
        bandPath = band.write(outDir)
        
        # Write the QA 
        pair.writeMask(outDir)