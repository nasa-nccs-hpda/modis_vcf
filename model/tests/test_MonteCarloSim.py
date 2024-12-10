
import logging
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from modis_vcf.model.MasterTraining import MasterTraining
from modis_vcf.model.MonteCarloSim import MonteCarloSim
from modis_vcf.model.Trial import Trial


# -----------------------------------------------------------------------------
# class MonteCarloSimTestCase
#
# python -W ignore -m unittest modis_vcf.model.tests.test_MonteCarloSim
# python -W ignore -m unittest modis_vcf.model.tests.test_MonteCarloSim.MonteCarloSimTestCase.testRunOneTrial   # noqa: E501
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
        self.assertEqual(mcs.numVarsForFinalModel, 20)
        self.assertEqual(mcs._minTimesEachVarUsed, 10)
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            numTrials = 1, 
                            predictorsPerTrial = 2, 
                            numVarsForFinalModel = 3,
                            minTimesEachVarUsed = 4,
                            logger=MonteCarloSimTestCase.logger)

        self.assertEqual(len(mcs.masterTraining.dataset.fragments), 2)
        self.assertEqual(mcs.numTrials, 1)
        self.assertEqual(mcs.predictorsPerTrial, 2)
        self.assertEqual(mcs.numVarsForFinalModel, 3)
        self.assertEqual(mcs._minTimesEachVarUsed, 4)

        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            numVarsForFinalModel = 3,
                            logger=MonteCarloSimTestCase.logger)

        self.assertEqual(len(mcs.masterTraining.dataset.fragments), 2)
        self.assertEqual(mcs.numTrials, 10)
        self.assertEqual(mcs.predictorsPerTrial, 10)
        self.assertEqual(mcs.numVarsForFinalModel, 3)
        self.assertEqual(mcs._minTimesEachVarUsed, 10)

    # -------------------------------------------------------------------------
    # testAllVars
    # -------------------------------------------------------------------------
    def testAllVars(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            logger=MonteCarloSimTestCase.logger)

        self.assertEqual(len(mcs.allVars), 255)
        
    # -------------------------------------------------------------------------
    # testAverages
    # -------------------------------------------------------------------------
    def testAverages(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            predictorsPerTrial = 4,
                            minTimesEachVarUsed = 1,
                            logger = MonteCarloSimTestCase.logger)

        trials: list = mcs._runTrials()
        averages: dict = mcs._computeAverages(trials)
        
        # Ensure all variables are represented in averages.
        self.assertEqual(len(mcs.allVars), len(averages))
        
        # ---
        # Ensure the variables used in the trials are exactly the ones reported
        # in the averages.
        # ---
        nonZeroAvgs: dict = {k:v for k, v in averages.items() if v != 0}
        usedVars = set([n for t in trials for n in t.predictorNames])
        self.assertEqual(nonZeroAvgs.keys(), usedVars)

        # Double check the non-zero averages in the trials.
        for predName in nonZeroAvgs:

            ims = [t.importanceMean(predName) 
                   for t in trials if t.includesPredictor(predName)]
            
            sumIm = sum(ims)
            avg = sumIm / len(ims)
            self.assertEqual(avg, averages[predName])
                 
    # -------------------------------------------------------------------------
    # testChooseColumns
    # -------------------------------------------------------------------------
    def testChooseColumns(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            predictorsPerTrial = 4,
                            logger = MonteCarloSimTestCase.logger)

        lastCols = None
        
        for i in range(100):

            curCols = mcs._chooseColumns()

            if not lastCols:
                lastCols = curCols
                
            elif curCols == lastCols:
                return False
                
        return True
        
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
        varUsageCount[firstKey] = 0
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
                            minTimesEachVarUsed = 0,
                            logger = MonteCarloSimTestCase.logger)
                 
        mcs.run()
        self.assertEqual(len(mcs.topN), mcs.numVarsForFinalModel)

    # -------------------------------------------------------------------------
    # testRunTrials
    # -------------------------------------------------------------------------
    def testRunTrials(self):

        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            predictorsPerTrial = 4,
                            logger = MonteCarloSimTestCase.logger)
                            
        trials: list[Trial] = mcs._runTrials()
        self.assertGreaterEqual(len(trials), mcs.numTrials)
        
        usedVars = set([n for t in trials for n in t.predictorNames])
        self.assertEqual(set(mcs.allVars), usedVars)
        
        # Ensure each trial has a different set of variables.
        allTrialPreds = np.asarray([t.predictorNames for t in trials])
        numUnique = np.unique(allTrialPreds, axis=0).shape[0]
        self.assertEqual(allTrialPreds.shape[0], numUnique)
        
    # -------------------------------------------------------------------------
    # testRunOneTrial
    # -------------------------------------------------------------------------
    def testRunOneTrial(self):

        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            predictorsPerTrial = 4,
                            logger = MonteCarloSimTestCase.logger)
                
        sampName = \
            mcs.masterTraining.dataset.schema.names[MasterTraining.SAMPLE_COL]
                        
        y: pd.DataFrame = \
             mcs.masterTraining.dataset.read([sampName]). \
             to_pandas().to_numpy().ravel()

        trial = mcs._runOneTrial(1)

    # -------------------------------------------------------------------------
    # testSaveFinalModel
    # -------------------------------------------------------------------------
    def testSaveFinalModel(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            predictorsPerTrial = 4,
                            minTimesEachVarUsed = 0,
                            logger = MonteCarloSimTestCase.logger)
             
        rf1 = mcs.run()
                       
        finalPath: Path = mcs.saveFinalModel(MonteCarloSimTestCase.base)
        print('Model File:', finalPath)
        
        with open(finalPath, 'rb') as f:
            rf2: RandomForestClassifier = pickle.load(f)

    # -------------------------------------------------------------------------
    # testTopN
    # -------------------------------------------------------------------------
    def testTopN(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.base,
                            predictorsPerTrial = 4,
                            minTimesEachVarUsed = 0,
                            logger = MonteCarloSimTestCase.logger)

        trials: list = mcs._runTrials()
        self.assertEqual(len(mcs.topN), mcs._numVarsForFinalModel)
        
        # ---
        # Ensure the top 1 has no average contribution greater than it.
        # Ensure the 2nd from top has only 1 contribution greater than it.
        # Etc.
        # ---
        averages: dict = mcs._computeAverages(trials)
        
        for i in range(mcs._numVarsForFinalModel - 1):
            
            curVal = mcs.topN[i][1]
            nextVal = mcs.topN[i+1][1]
            self.assertGreater(curVal, nextVal)
            
        