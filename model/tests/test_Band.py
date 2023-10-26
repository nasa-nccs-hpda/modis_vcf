
import filecmp
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from modis_vcf.model.Band import Band


# -----------------------------------------------------------------------------
# class BandTestCase
#
# python -m unittest modis_vcf.model.tests.test_Band
# python -m unittest modis_vcf.model.tests.test_Band.BandTestCase.testReadWrite
# -----------------------------------------------------------------------------
class BandTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        band = Band()
        
        cube = np.zeros(shape=(12, Band.ROWS, Band.COLS))
        band = Band('testBand', cube, {})

    # -------------------------------------------------------------------------
    # testCreateDs
    # -------------------------------------------------------------------------
    def testCreateDs(self):
        
        band = Band()
        outName = Path(tempfile.mkdtemp()) / 'test.tif'

        with self.assertRaisesRegex(RuntimeError, 'Cannot create a data set'):
            band.createDs(outName)

        cube = np.zeros(shape=(12, 10, 10))
        band = Band('testBand', cube, {})
        band.createDs(outName)
        os.remove(outName)
            
    # -------------------------------------------------------------------------
    # testWrite
    # -------------------------------------------------------------------------
    def testWrite(self):
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        numBands = 23
        shape = (numBands, Band.ROWS, Band.COLS)
        rng = np.random.default_rng()
        cube = (10000 * rng.random(size=shape)).astype(Band.NUMPY_DTYPE)
        
        xref = {}

        for i in range(numBands):
            xref[str(i)] = i
        
        band = Band('JunkBand', cube, xref, logger)
        
        outDir = Path(tempfile.mkdtemp())
        band.write(outDir)     
        
    # -------------------------------------------------------------------------
    # testReadWrite
    # -------------------------------------------------------------------------
    def testReadWrite(self):
        
        # Write
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        numBands = 23
        shape = (numBands, Band.ROWS, Band.COLS)
        rng = np.random.default_rng()
        cube = (10000 * rng.random(size=shape)).astype(Band.NUMPY_DTYPE)
        
        xref = {}

        for i in range(numBands):
            xref[str(i)] = i
        
        outBand = Band('JunkBand', cube, xref, logger)
        
        outDir = Path(tempfile.mkdtemp())
        outName = outBand.write(outDir)     

        # Read
        inBand = Band()
        inBand.read(outName)
        
        self.assertTrue((outBand.cube == inBand.cube).all())
