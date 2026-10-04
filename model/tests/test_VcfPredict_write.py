from pathlib import Path
import tempfile
import unittest

import numpy as np
from osgeo import gdal

from modis_vcf.model.VcfPredict import VcfPredict


# -----------------------------------------------------------------------------
# class VcfPredictWriteTestCase
#
# Checks the output storage types (uint8 percent layers with fill 253, uint16 SD layers) on small synthetic rasters,
# so it needs no tiles, metrics or model files.
#
# python -m unittest modis_vcf.model.tests.test_VcfPredict_write
# -----------------------------------------------------------------------------
class VcfPredictWriteTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUp
    # -------------------------------------------------------------------------
    def setUp(self):

        self._tmp = tempfile.TemporaryDirectory()
        self._vcfp = VcfPredict.__new__(VcfPredict)
        self._vcfp._outDir = Path(self._tmp.name)

    # -------------------------------------------------------------------------
    # tearDown
    # -------------------------------------------------------------------------
    def tearDown(self):

        self._tmp.cleanup()

    # -------------------------------------------------------------------------
    # testPercentLayerIsUint8
    # -------------------------------------------------------------------------
    def testPercentLayerIsUint8(self):

        raster = np.array([[0, 37, 100], [200, 250, 5]], dtype=np.int16)
        outName = self._vcfp._write('h09v05', 2019, 'Percent_Tree_Cover', raster)

        ds = gdal.Open(str(outName))
        band = ds.GetRasterBand(1)
        self.assertEqual(band.DataType, gdal.GDT_Byte)
        self.assertEqual(band.GetNoDataValue(), VcfPredict.PCT_NO_DATA)
        self.assertTrue(np.array_equal(band.ReadAsArray(), raster))

    # -------------------------------------------------------------------------
    # testSdLayerIsUint16
    # -------------------------------------------------------------------------
    def testSdLayerIsUint16(self):

        raster = np.array([[0, 3632, 5000], [12, 0, 1]], dtype=np.int16)
        outName = self._vcfp._write('h09v05', 2019, 'Percent_Tree_Cover_SD', raster, VcfPredict.SD_DATA_TYPE)

        ds = gdal.Open(str(outName))
        band = ds.GetRasterBand(1)
        self.assertEqual(band.DataType, gdal.GDT_UInt16)
        self.assertIsNone(band.GetNoDataValue())
        self.assertTrue(np.array_equal(band.ReadAsArray(), raster))

    # -------------------------------------------------------------------------
    # testOutOfRangeRefused
    # -------------------------------------------------------------------------
    def testOutOfRangeRefused(self):

        with self.assertRaises(ValueError):
            self._vcfp._write('h09v05', 2019, 'Percent_Tree_Cover', np.array([[-1, 300]], dtype=np.int16))

        with self.assertRaises(ValueError):
            self._vcfp._write('h09v05', 2019, 'Percent_Tree_Cover_SD', np.array([[-1, 0]], dtype=np.int16),
                              VcfPredict.SD_DATA_TYPE)
