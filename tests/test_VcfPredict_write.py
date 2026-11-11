from pathlib import Path
import tempfile
import unittest

import numpy as np
from osgeo import gdal
from osgeo import osr

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

    # -------------------------------------------------------------------------
    # testModisSinusoidalGrid
    #
    # Expected corner and pixel size from the C6.1 MOD44B h08v04 granule (MOD44B.A2019065.h08v04.061).
    # -------------------------------------------------------------------------
    def testModisSinusoidalGrid(self):

        raster = np.zeros((4800, 4800), dtype=np.int16)
        outName = self._vcfp._write('h08v04', 2019, 'Percent_Tree_Cover', raster)

        ds = gdal.Open(str(outName))
        expected = (-11119505.196667, 231.656358263958, 0.0, 5559752.598333, 0.0, -231.656358263958)

        for got, exp in zip(ds.GetGeoTransform(), expected):
            self.assertAlmostEqual(got, exp, places=5)

        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        self.assertEqual(srs.GetSemiMajor(), 6371007.181)
        self.assertEqual(srs.GetSemiMinor(), 6371007.181)
