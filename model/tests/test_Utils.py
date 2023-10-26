
import logging
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from osgeo import gdal
from osgeo.osr import SpatialReference

from modis_vcf.model.Utils import Utils


# -----------------------------------------------------------------------------
# class UtilsTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_Utils
# python -m unittest modis_vcf.model.tests.test_Utils.UtilsTestCase.testCreateDsFromParams
# -----------------------------------------------------------------------------
class UtilsTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # testCreateDsFromParams
    # -------------------------------------------------------------------------
    def testCreateDsFromParams(self):
        
        outDir = Path(tempfile.mkdtemp())
        outName = outDir / 'test.tif'
        shape = (2, 3)
        numBands = 4
        dataType = gdal.GDT_Int16
        
        Utils.createDsFromParams(outName, shape, numBands, dataType)
        outName.unlink()
        outDir.rmdir()
        
    # -------------------------------------------------------------------------
    # testCreateDs
    # -------------------------------------------------------------------------
    def testCreateDs(self):

        cube = np.full((2, 2, 3), np.nan)
        cube[0] = [[1, 2, 3], [4, 5, 6]]
        cube[1] = [[7, 8, 9], [10, 11, 12]]

        dayXref = {'211200': 0, '100100': 1}
        dataType = gdal.GDT_Int16
        band = Utils.Band('testBand', dataType, cube, dayXref)

        outDir = Path(tempfile.mkdtemp())
        outName = outDir / 'test.tif'
        Utils.createDs(outName, band)
        outName.unlink()
        outDir.rmdir()
        
    # -------------------------------------------------------------------------
    # testWriteBand
    # -------------------------------------------------------------------------
    def testWriteBand(self):
        
        cube = np.full((2, 2, 3), np.nan)
        cube[0] = [[1, 2, 3], [4, 5, 6]]
        cube[1] = [[7, 8, 9], [10, 11, 12]]

        dayXref = {'211200': 0, '100100': 1}
        dataType = gdal.GDT_Int16
        band = Utils.Band('testBand', dataType, cube, dayXref)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        outDir = Path(tempfile.mkdtemp())
        outName = outDir / 'test.tif'
        outFile = Utils.writeBand(outDir, band, logger)
        outFile.unlink()
        outDir.rmdir()
        