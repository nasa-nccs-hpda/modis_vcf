
import random
import unittest

from modis_vcf.model.Trial import Trial


# -----------------------------------------------------------------------------
# class TrialTestCase
#
# python -m unittest modis_vcf.model.tests.test_Trial
# python -m unittest modis_vcf.model.tests.test_Trial.TrialTestCase.testInit
# -----------------------------------------------------------------------------
class TrialTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUpClass
    # -------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):

        pts1D = random.choices(range(100), k = 5)
        cls._sampleLocs = [(int(p / 5), p % 5) for p in pts1D]

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        t = Trial('test', TrialTestCase._sampleLocs)
        self.assertEqual(len(t._sampleLocs), 5)
        self.assertEqual(len(t._sampleLocs[0]), 2)

    # -------------------------------------------------------------------------
    # testMinMaxRowCol
    # -------------------------------------------------------------------------
    def testMinMaxRowCol(self):
        
        sampleLocs = [(6, 2), (0, 3), (9, 2), (14, 2), (14, 4)]
        t = Trial('test', sampleLocs)
        self.assertEqual(t.minRow(), 0)
        self.assertEqual(t.minCol(), 2)
        self.assertEqual(t.maxRow(), 14)
        self.assertEqual(t.maxCol(), 4)
        
    # -------------------------------------------------------------------------
    # testSampleLocs
    # -------------------------------------------------------------------------
    def testSampleLocs(self):

        t = Trial('test', TrialTestCase._sampleLocs)
        self.assertEqual(len(t.sampleLocs), 5)
        self.assertEqual(len(t.sampleLocs[0]), 2)
