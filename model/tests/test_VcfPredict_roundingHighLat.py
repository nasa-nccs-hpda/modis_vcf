import logging
import tempfile
import unittest
from unittest import mock

import numpy as np

from modis_vcf.model.VcfPredict import VcfPredict


# -----------------------------------------------------------------------------
# class VcfPredictRoundingHighLatTestCase
#
# Checks rounding of the value cast and the v00 high-latitude rule on a 2 x 2
# tile, with the predictions and the water mask stubbed, so it needs no tiles,
# metrics or model files.
#
# python -m unittest modis_vcf.model.tests.test_VcfPredict_roundingHighLat
# -----------------------------------------------------------------------------
class VcfPredictRoundingHighLatTestCase(unittest.TestCase):

    # Land, water, land, out of projection.
    MASK = np.array([[0, VcfPredict.MASK_WATER], [0, VcfPredict.MASK_OUT_OF_PROJ]])

    # -------------------------------------------------------------------------
    # runTile
    # -------------------------------------------------------------------------
    def _runTile(self, tid):

        vp = VcfPredict.__new__(VcfPredict)
        vp._logger = logging.getLogger('test')
        vp._waterMasks = {tid + '2019': VcfPredictRoundingHighLatTestCase.MASK}
        # Stand-in models (_runPrediction is mocked); None would mean "no models" (rule tiles only).
        vp._treeCoverRf = vp._nonvegRf = object()
        vp._treeCoverTopN = vp._nonvegTopN = None
        vp._partnerRf = None   # no blend A
        vp._bp10 = False       # the rules under test only
        vp._treeCoverAgg = vp._nonvegAgg = 'mean'
        vp._outDir = self._tmp.name     # <tile>-<year>-chain.json
        vp._productType = vp._metricsDir = None

        tc = vp._maskPrediction(tid, 2019, np.array([[30, 0], [70, 0]], dtype=np.int16))
        nv = vp._maskPrediction(tid, 2019, np.array([[50, 0], [40, 0]], dtype=np.int16))
        sd = np.array([[500, 0], [700, 0]], dtype=np.int16)
        written = {}
        flags = mock.Mock()
        flags.return_value.getPeriodFlags.return_value = (np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8))

        with mock.patch.object(VcfPredict, '_runPrediction', side_effect=[(tc, sd), (nv, sd)]), \
             mock.patch.object(VcfPredict, '_write',
                               side_effect=lambda t, y, name, a, *args, **kw: written.__setitem__(name, a)), \
             mock.patch('modis_vcf.model.VcfPredict.Metrics', flags):   # Quality / Cloud read no metrics

            vp.runTileForYear(tid, 2019)

        return written

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    # -------------------------------------------------------------------------
    # testRoundValue
    # -------------------------------------------------------------------------
    def testRoundValue(self):

        v = np.array([0.0, 0.49, 0.5, 1.5, 2.4999, 99.5, 100.0])
        out = VcfPredict._roundValue(v)
        np.testing.assert_array_equal(out, [0, 0, 1, 2, 2, 100, 100])
        self.assertEqual(out.dtype, np.int16)

    # -------------------------------------------------------------------------
    # testVTile
    # -------------------------------------------------------------------------
    def testVTile(self):

        self.assertEqual(VcfPredict._vTile('h16v00'), 0)
        self.assertEqual(VcfPredict._vTile('h12v09'), 9)
        self.assertEqual(VcfPredict._vTile('h31v11'), 11)

    # -------------------------------------------------------------------------
    # testHighLatitudeRule
    # -------------------------------------------------------------------------
    def testHighLatitudeRule(self):

        w = self._runTile('h17v00')

        # Land is tree 0, non-vegetated 100, non-tree vegetation 0; codes are kept.
        np.testing.assert_array_equal(w['Percent_Tree_Cover'], [[0, 200], [0, 250]])
        np.testing.assert_array_equal(w['Percent_NonVegetated'], [[100, 200], [100, 250]])
        np.testing.assert_array_equal(w['Percent_NonTree_Vegetation'], [[0, 200], [0, 250]])

        # The SD layers are untouched.
        np.testing.assert_array_equal(w['Percent_Tree_Cover_SD'], [[500, 0], [700, 0]])

    # -------------------------------------------------------------------------
    # testOtherTilesUnchanged
    # -------------------------------------------------------------------------
    def testOtherTilesUnchanged(self):

        w = self._runTile('h17v01')

        np.testing.assert_array_equal(w['Percent_Tree_Cover'], [[30, 200], [70, 250]])

        # Sum cap: non-vegetated 40 is reduced to 100 - 70 = 30.
        np.testing.assert_array_equal(w['Percent_NonVegetated'], [[50, 200], [30, 250]])
        np.testing.assert_array_equal(w['Percent_NonTree_Vegetation'], [[20, 200], [0, 250]])
