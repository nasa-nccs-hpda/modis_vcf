
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

        NAME = 'test'
        PRED_NAMES = ['pred1', 'pred2']
        PREM_IMPORT =  {'a': 1, 'b': 2}
        t = Trial(NAME, PRED_NAMES, PREM_IMPORT)
        self.assertEqual(t.name, NAME)
        self.assertEqual(t.predictorNames, PRED_NAMES)
        self.assertEqual(t.permImportance, PREM_IMPORT)
        