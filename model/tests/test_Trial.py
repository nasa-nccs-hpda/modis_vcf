
from pathlib import Path
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

        cls.importance = {'importances_mean': [1, 2], 
                          'importances_std': [3, 4],
                          'importances': []}
       
    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        NAME = 'test'
        PRED_NAMES = ['pred1', 'pred2']
        t = Trial(NAME, PRED_NAMES, TrialTestCase.importance)
        self.assertEqual(t.name, NAME)
        self.assertEqual(t.predictorNames, PRED_NAMES)
        self.assertEqual(t.permImportances, TrialTestCase.importance)

    # -------------------------------------------------------------------------
    # testIncludesPredictor
    # -------------------------------------------------------------------------
    def testIncludesPredictor(self):

        NAME = 'test'
        PRED_NAMES = ['pred1', 'pred2']
        t = Trial(NAME, PRED_NAMES, TrialTestCase.importance)
        self.assertTrue(t.includesPredictor('pred1'))
        self.assertTrue(t.includesPredictor('pred2'))
        self.assertFalse(t.includesPredictor('pred3'))

    # -------------------------------------------------------------------------
    # testIndex
    # -------------------------------------------------------------------------
    def testIndex(self):

        NAME = 'test'
        PRED_NAMES = ['pred1', 'pred2']
        t = Trial(NAME, PRED_NAMES, TrialTestCase.importance)
        self.assertEqual(t.index('pred1'), 0)
        self.assertEqual(t.index('pred2'), 1)
        self.assertEqual(t.index('pred3'), -1)

    # -------------------------------------------------------------------------
    # testImportanceMean
    # -------------------------------------------------------------------------
    def testImportanceMean(self):

        NAME = 'test'
        PRED_NAMES = ['pred1', 'pred2']
        t = Trial(NAME, PRED_NAMES, TrialTestCase.importance)
        self.assertEqual(t.importanceMean('pred1'), 1)
        self.assertEqual(t.importanceMean('pred2'), 2)
        
        with self.assertRaisesRegex(RuntimeError, 'oes not include predictor'):
            t.importanceMean('nope')

    # -------------------------------------------------------------------------
    # testPersistence
    # -------------------------------------------------------------------------
    def testPersistence(self):
    
        NAME = 'test'
        PRED_NAMES = ['pred1', 'pred2']
        outTrial = Trial(NAME, PRED_NAMES, TrialTestCase.importance)
        outPath = Path(__file__).parent 
        outFile = outTrial.save(outPath)
        self.assertTrue(outFile.exists())

        inTrial = Trial.load(outFile)
        self.assertEqual(outTrial._name, inTrial._name)
        self.assertEqual(outTrial._permImportances, inTrial._permImportances)
        self.assertEqual(outTrial._predictorNames, inTrial._predictorNames)
    