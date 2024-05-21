
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
    # testAllVars
    # -------------------------------------------------------------------------
    def testAllVars(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            logger=MonteCarloSimTestCase.logger)

        self.assertEqual(len(mcs.allVars), 255)
        
    # -------------------------------------------------------------------------
    # testPollVarUsage
    # -------------------------------------------------------------------------
    def testPollVarUsage(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            logger=MonteCarloSimTestCase.logger)

        # Simulate the usage count.
        varUsageCount = dict.fromkeys(mcs.allVars, 0) 
        self.assertEqual(len(varUsageCount), 255)
        self.assertTrue(all(v == 0 for v in varUsageCount.values()))
        self.assertFalse(mcs._pollVarUsage(varUsageCount))
        
        # Fill all variables with the minimum value.
        keys = list(varUsageCount.keys())
        firstKey = keys[0]
        
        varUsageCount = {k: varUsageCount[k] + mcs._minTimesEachVarUsed \
                         for k in keys}
                         
        self.assertTrue(mcs._pollVarUsage(varUsageCount))

        # Make one variable insuffcient.
        varUsageCount[firstKey] = mcs._minTimesEachVarUsed - 1
        self.assertFalse(mcs._pollVarUsage(varUsageCount))
        
        # Test too few variables.
        varUsageCount[firstKey] = mcs._minTimesEachVarUsed
        self.assertTrue(mcs._pollVarUsage(varUsageCount))
        del varUsageCount[firstKey]
        self.assertFalse(mcs._pollVarUsage(varUsageCount))
                         
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
    # testRun
    # -------------------------------------------------------------------------
    def testRun(self):

        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            predictorsPerTrial = 4,
                            logger = MonteCarloSimTestCase.logger)
                            
        mcs.run()

    # -------------------------------------------------------------------------
    # testRunTrials
    # -------------------------------------------------------------------------
    def testRunTrials(self):

        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            predictorsPerTrial = 4,
                            logger = MonteCarloSimTestCase.logger)
                            
        trials = mcs._runTrials()
        
        # Ensure every variable was used.
        usedVars = set()
        
        for trial in trials:
            usedVars.update(trial.predictorNames)
        
        self.assertEqual(set(mcs.allVars), usedVars)
        
    # -------------------------------------------------------------------------
    # testRunOneTrial
    # -------------------------------------------------------------------------
    def testRunOneTrial(self):

        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            predictorsPerTrial = 4,
                            logger = MonteCarloSimTestCase.logger)
                
        trial = mcs._runOneTrial(1)
        