
import logging
from pathlib import Path
import sys
import unittest

from modis_vcf.model.MasterTraining import MasterTraining
from modis_vcf.model.MonteCarloSimGpu import MonteCarloSimGpu


# -----------------------------------------------------------------------------
# class MonteCarloSimTestCase
#
# python -m unittest modis_vcf.model.tests.test_MonteCarloSimGpu
# python -W ignore -m unittest modis_vcf.model.tests.test_MonteCarloSimGpu.MonteCarloSimGpuTestCase.testInit   # noqa: E501
# -----------------------------------------------------------------------------
class MonteCarloSimGpuTestCase(unittest.TestCase):

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

        mcs = MonteCarloSimGpu(MonteCarloSimGpuTestCase.base,
                              logger=MonteCarloSimGpuTestCase.logger)

        self.assertEqual(len(mcs.masterTraining.dataset.fragments), 2)
        self.assertEqual(mcs.numTrials, 10)
        self.assertEqual(mcs.predictorsPerTrial, 10)
        self.assertEqual(mcs._numVarsForFinalModel, 20)
        self.assertEqual(mcs._minTimesEachVarUsed, 10)
        
        mcs = MonteCarloSimGpu(MonteCarloSimGpuTestCase.base,
                               numTrials = 1, 
                               predictorsPerTrial = 2, 
                               numVarsForFinalModel = 3,
                               minTimesEachVarUsed = 4,
                               logger=MonteCarloSimGpuTestCase.logger)

        self.assertEqual(len(mcs.masterTraining.dataset.fragments), 2)
        self.assertEqual(mcs.numTrials, 1)
        self.assertEqual(mcs.predictorsPerTrial, 2)
        self.assertEqual(mcs._numVarsForFinalModel, 3)
        self.assertEqual(mcs._minTimesEachVarUsed, 4)

        mcs = MonteCarloSimGpu(MonteCarloSimGpuTestCase.base,
                               numVarsForFinalModel = 3,
                               logger=MonteCarloSimGpuTestCase.logger)

        self.assertEqual(len(mcs.masterTraining.dataset.fragments), 2)
        self.assertEqual(mcs.numTrials, 10)
        self.assertEqual(mcs.predictorsPerTrial, 10)
        self.assertEqual(mcs._numVarsForFinalModel, 3)
        self.assertEqual(mcs._minTimesEachVarUsed, 10)

    # -------------------------------------------------------------------------
    # testRunOneTrial
    # -------------------------------------------------------------------------
    def testRunOneTrial(self):

        mcs = MonteCarloSimGpu(MonteCarloSimGpuTestCase.base,
                               predictorsPerTrial = 4,
                               logger = MonteCarloSimGpuTestCase.logger)
                
        sampName = \
            mcs.masterTraining.dataset.schema.names[MasterTraining.SAMPLE_COL]
                        
        y: pd.DataFrame = \
             mcs.masterTraining.dataset.read([sampName]). \
             to_pandas().to_numpy().ravel()

        trial = mcs._runOneTrial(1)

