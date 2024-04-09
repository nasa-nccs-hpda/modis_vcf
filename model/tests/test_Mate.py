
import logging
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from osgeo import gdal

from modis_vcf.model.Mate import Mate


# -----------------------------------------------------------------------------
# class MateTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_Mate
# python -m unittest modis_vcf.model.tests.test_Mate.MateTestCase.testRead
# -----------------------------------------------------------------------------
class MateTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        # Test with no arguments.
        mate = Mate()
        self.assertIsNone(mate._fileName)
        self.assertIsNone(mate._bandType)
        self.assertIsNone(mate._dataset)
        
        # Test file name.
        fn = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')

        mate = Mate(fn)
        self.assertEqual(mate._fileName, fn)
        
    # -------------------------------------------------------------------------
    # testDiscoverBandType
    # -------------------------------------------------------------------------
    def testDiscoverBandType(self):
        
        fn = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')

        mate = Mate(fn)
        self.assertEqual(mate.bandType, Mate.BAND_TYPES[Mate.CH])
        
    # -------------------------------------------------------------------------
    # testBandType
    # -------------------------------------------------------------------------
    def testBandType(self):
        
        mate = Mate()
        mate.bandType = Mate.BAND_TYPES[Mate.CH]
        self.assertEqual(mate.bandType.code, Mate.CH)
        self.assertEqual(mate.bandType.index, 0)

        with self.assertRaisesRegex(RuntimeError, 'Invalid band type, Hi'):
            bt = Mate.BandType('Hi', 2112)
            mate.bandType = bt
            
    # -------------------------------------------------------------------------
    # testDataset
    # -------------------------------------------------------------------------
    def testDataset(self):
        
        fn = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')
                  
        ds = gdal.Open(str(fn))
        mate = Mate(fn)
        self.assertEqual(mate.dataset.GetFileList(), ds.GetFileList())
        
    # -------------------------------------------------------------------------
    # testFileName
    # -------------------------------------------------------------------------
    def testFileName(self):
        
        fn = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')

        mate = Mate()
        mate.fileName = fn
        self.assertEqual(mate.fileName, fn)

    # -------------------------------------------------------------------------
    # testDiscoverBandType
    # -------------------------------------------------------------------------
    def testDiscoverBandType(self):
        
        fn = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/' \
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')

        self.assertEqual(Mate.discoverBandType(fn), Mate.BAND_TYPES[Mate.CH])
        
    # -------------------------------------------------------------------------
    # testMyGetKey
    # -------------------------------------------------------------------------
    def testMyGetKey(self):
        
        ch = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/'
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')

        mate = Mate(ch)
        self.assertEqual(mate.getMyKey(), '2019065')        

    # -------------------------------------------------------------------------
    # testRead
    # -------------------------------------------------------------------------
    def testRead(self):
        
        ch = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/'
                  'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')

        mate = Mate(ch)
        band, dataType = mate.read(1)
        
        self.assertEqual(band.shape, (4800, 4800))
        self.assertEqual(dataType, gdal.GDT_Int16)

    # -------------------------------------------------------------------------
    # testValidity
    # -------------------------------------------------------------------------
    # def testValidity(self):
    #
    #     ch = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C/'
    #               'MOD44CH.A2019065.h09v05.061.2020290183523.hdf')
    #
    #     mate = Mate(ch)
    #     band, dataType = mate.read(1)
    #     outName = Path(tempfile.mkdtemp()) / '2020290183523.tif'
    #     print('Writing', outName)
    #     ds = Utils.createDsFromParams(outName, band.shape, 1, dataType)
    #     gdBand = ds.GetRasterBand(1)
    #     gdBand.WriteArray(band)
    #     gdBand.FlushCache()
    #     gdBand = None
    #     ds = None
        
        
        