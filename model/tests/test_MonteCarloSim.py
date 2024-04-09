
import logging
from pathlib import Path
import sys
import unittest

from modis_vcf.model.MonteCarloSim import MonteCarloSim
from modis_vcf.model.Trial import Trial


# -----------------------------------------------------------------------------
# class MonteCarloSimTestCase
#
# python -m unittest modis_vcf.model.tests.test_MonteCarloSim
# python -m unittest modis_vcf.model.tests.test_MonteCarloSim.MonteCarloSimTestCase.testInit
# -----------------------------------------------------------------------------
class MonteCarloSimTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUpClass
    # -------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):

        cls.logger = logging.getLogger()
        cls.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        cls.logger.addHandler(ch)

        cls.base = Path(__file__).parent

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            MonteCarloSimTestCase.logger)

        self.assertEqual(len(mcs._trainingDs.dataset.fragments), 2)
        self.assertEqual(mcs._trainingDs.numCols, 259 - 4 + 259 - 4)
        self.assertEqual(mcs._trainingDs.numRows, 536226 + 268039)

    # -------------------------------------------------------------------------
    # testMasterTraining
    # -------------------------------------------------------------------------
    def testMasterTraining(self):

        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            MonteCarloSimTestCase.logger)
          
        self.assertEqual(len(mcs.masterTraining.dataset.fragments), 2)
        
    # -------------------------------------------------------------------------
    # testProperties
    # -------------------------------------------------------------------------
    def testProperties(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            MonteCarloSimTestCase.logger)
        
        self.assertEqual(mcs.numTrials, 10)
        self.assertEqual(mcs.predictorsPerTrial, 10)
        
    # -------------------------------------------------------------------------
    # testRandomizedSelection
    # -------------------------------------------------------------------------
    def testRandomizedSelection(self):

        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            MonteCarloSimTestCase.logger)
                     
        trials: list = mcs._randomizedSelection()
        self.assertEqual(len(trials), mcs._numTrials)
        self.assertEqual(len(trials[0]._sampleLocs), mcs._predictorsPerTrial)
        self.assertEqual(len(trials[0]._sampleLocs[0]), 2)
        
        # Test the range of x and y.
        for trial in trials:

            self.assertGreaterEqual(trial.minRow(), 0)
            self.assertLess(trial.minRow(), mcs.masterTraining.numRows)
            self.assertGreaterEqual(trial.minCol(), 0)
            self.assertLess(trial.minCol(), mcs.masterTraining.numCols)

