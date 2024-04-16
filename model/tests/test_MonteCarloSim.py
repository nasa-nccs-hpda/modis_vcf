
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
# python -m unittest modis_vcf.model.tests.test_MonteCarloSim.MonteCarloSimTestCase.testRunOneTrial
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
                            logger=MonteCarloSimTestCase.logger)

        self.assertEqual(len(mcs.masterTraining.dataset.fragments), 2)
        self.assertEqual(mcs.numTrials, 10)
        self.assertEqual(mcs.predictorsPerTrial, 10)

    # -------------------------------------------------------------------------
    # testProperties
    # -------------------------------------------------------------------------
    def testProperties(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            logger=MonteCarloSimTestCase.logger)
        
        self.assertEqual(len(mcs.masterTraining.dataset.fragments), 2)
        self.assertEqual(mcs.numTrials, 10)
        self.assertEqual(mcs.predictorsPerTrial, 10)
        
    # -------------------------------------------------------------------------
    # testRunOneTrial
    # -------------------------------------------------------------------------
    def testRunOneTrial(self):

        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            predictorsPerTrial = 4,
                            logger = MonteCarloSimTestCase.logger)
                            
        mcs._runOneTrial(1)
