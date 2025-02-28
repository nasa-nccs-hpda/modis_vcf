
import logging
import multiprocessing
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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

        cls.trainingDir = Path(__file__).parent
        cls.outDir = Path(tempfile.mkdtemp())
        cls.outDir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                            MonteCarloSimTestCase.outDir,
                            procInputTrainFilesIndependently = False,
                            logger=MonteCarloSimTestCase.logger)

        self.assertEqual(len(mcs._masterTraining.dataset.fragments), 2)
        self.assertEqual(mcs._predictorsPerTrial, 10)
        self.assertEqual(mcs.numVarsForFinalModel, 20)
        self.assertEqual(mcs._minTimesEachVarUsed, 10)
        self.assertEqual(mcs._numCpus, 1)
        self.assertIsInstance(mcs._X, pd.core.frame.DataFrame)
        self.assertIsInstance(mcs._xTrain, pd.core.frame.DataFrame)
        self.assertIsInstance(mcs._xTest, pd.core.frame.DataFrame)
        self.assertIsInstance(mcs._yTrain, pd.Series)
        self.assertIsInstance(mcs._yTest, pd.Series)
        self.assertIsInstance(mcs._y, pd.Series)
        self.assertEqual(len(mcs._X), len(mcs._xTrain) + len(mcs._xTest))
        self.assertEqual(len(mcs._y), len(mcs._yTrain) + len(mcs._yTest))
        self.assertEqual(mcs._X.shape, (2000, 258))
        self.assertEqual(mcs._y.shape, (2000,))
        
        # The number of trials and number of CPUs should be adjusted.
        mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                            MonteCarloSimTestCase.outDir,
                            predictorsPerTrial = 2, 
                            numVarsForFinalModel = 3,
                            minTimesEachVarUsed = 4,
                            numCpus = 2112,
                            procInputTrainFilesIndependently = False,
                            logger=MonteCarloSimTestCase.logger)

        self.assertEqual(len(mcs._masterTraining.dataset.fragments), 2)
        self.assertEqual(mcs._predictorsPerTrial, 2)
        self.assertEqual(mcs.numVarsForFinalModel, 3)
        self.assertEqual(mcs._minTimesEachVarUsed, 4)
        self.assertEqual(mcs._numCpus, multiprocessing.cpu_count())
        self.assertIsInstance(mcs._X, pd.core.frame.DataFrame)
        self.assertIsInstance(mcs._xTrain, pd.core.frame.DataFrame)
        self.assertIsInstance(mcs._xTest, pd.core.frame.DataFrame)
        self.assertIsInstance(mcs._yTrain, pd.Series)
        self.assertIsInstance(mcs._yTest, pd.Series)
        self.assertIsInstance(mcs._y, pd.Series)
        self.assertEqual(len(mcs._X), len(mcs._xTrain) + len(mcs._xTest))
        self.assertEqual(len(mcs._y), len(mcs._yTrain) + len(mcs._yTest))
        self.assertEqual(mcs._X.shape, (2000, 258))
        self.assertEqual(mcs._y.shape, (2000,))

        mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                            MonteCarloSimTestCase.outDir,
                            numVarsForFinalModel = 3,
                            logger=MonteCarloSimTestCase.logger)

        self.assertEqual(len(mcs._masterTraining.dataset.fragments), 2)
        self.assertEqual(mcs._predictorsPerTrial, 10)
        self.assertEqual(mcs.numVarsForFinalModel, 3)
        self.assertEqual(mcs._minTimesEachVarUsed, 10)
        self.assertEqual(mcs._numCpus, 1)
        self.assertIsInstance(mcs._X, pd.core.frame.DataFrame)
        self.assertIsInstance(mcs._xTrain, pd.core.frame.DataFrame)
        self.assertIsInstance(mcs._xTest, pd.core.frame.DataFrame)
        self.assertIsInstance(mcs._yTrain, pd.Series)
        self.assertIsInstance(mcs._yTest, pd.Series)
        self.assertIsInstance(mcs._y, pd.Series)
        self.assertEqual(len(mcs._X), len(mcs._xTrain) + len(mcs._xTest))
        self.assertEqual(len(mcs._y), len(mcs._yTrain) + len(mcs._yTest))
        self.assertEqual(mcs._X.shape, (2000, 258))
        self.assertEqual(mcs._y.shape, (2000,))

    # -------------------------------------------------------------------------
    # testAllVars
    # -------------------------------------------------------------------------
    def testAllVars(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                            MonteCarloSimTestCase.outDir,
                            logger=MonteCarloSimTestCase.logger)

        self.assertEqual(len(mcs._allVars), 255)
        
    # -------------------------------------------------------------------------
    # testChooseColumns
    # -------------------------------------------------------------------------
    def testChooseColumns(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                            MonteCarloSimTestCase.outDir,
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
    # testComputeAverages
    # -------------------------------------------------------------------------
    def testComputeAverages(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                            MonteCarloSimTestCase.outDir,
                            predictorsPerTrial = 4,
                            minTimesEachVarUsed = 0,
                            logger = MonteCarloSimTestCase.logger)

        mcs.run()
        averages: dict = mcs.computeAverages()
        
        # Ensure all variables are represented in averages.
        self.assertEqual(len(averages), len(mcs._allVars))
        self.assertEqual(set(list(averages.keys())), set(mcs._allVars))
        
        # ---
        # Check a specific average.  First, find a predictor with a non-zero
        # average.
        # ---
        nonZeroVar: str = None
        
        for var, avg in averages.items():
            
            if avg != 0:
                
                nonZeroVar = var
                break
                
        self.assertIsNotNone(nonZeroVar)
        
        # Collect the permutation importance for that variable.
        nonZeroVarPermImports = []
        
        for trial in mcs._trials:
                
            if var in trial.predictorNames:
                
                index = trial.predictorNames.index(var)
                
                pImport = \
                    trial.permImportances['importances_mean'][index]

                if pImport != 0:
                    nonZeroVarPermImports.append(pImport)
                    
        # Compute its average.
        avg = sum(nonZeroVarPermImports) / len(nonZeroVarPermImports)
        self.assertEqual(avg, averages[nonZeroVar])

    # -------------------------------------------------------------------------
    # testGetTopN1
    # -------------------------------------------------------------------------
    def testGetTopN1(self):

        # ---
        # This case covers the scenario where the number of predictors
        # specified per trial (1) and the number of trials (10) do not 
        # involve enough predictors to satisfy the number of predictors for
        # the final model (20).  MCS warns and adjusts until it satisfied the
        # conditions.
        # ---
        mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                            MonteCarloSimTestCase.outDir,
                            predictorsPerTrial = 1,
                            numVarsForFinalModel = 20,
                            minTimesEachVarUsed = 0,
                            logger = MonteCarloSimTestCase.logger)

        mcs.run()

    # -------------------------------------------------------------------------
    # testGetTopN2
    # -------------------------------------------------------------------------
    def testGetTopN2(self):

        with warnings.catch_warnings():
            
            # warnings.simplefilter('ignore')

            mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                                MonteCarloSimTestCase.outDir,
                                minTimesEachVarUsed = 1,
                                logger = MonteCarloSimTestCase.logger)

            try:
                mcs.run()

            except RuntimeError as e:

                # This should not happen because minTimesEachVarUsed is 1.
                if str(e).startswith('Some top-n columns were not used in'):
                    
                    assert False, 'Some top-n cols. were not used in trials.'
                    
                else:
                    raise e
                    
    # -------------------------------------------------------------------------
    # testMinVarUsageAchieved
    # -------------------------------------------------------------------------
    def testMinVarUsageAchieved(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                            MonteCarloSimTestCase.outDir,
                            logger=MonteCarloSimTestCase.logger)

        self.assertFalse(mcs.minVarUsageAchieved())
        
        # Verify minTimesEachVarUsed = 1. 
        mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                            MonteCarloSimTestCase.outDir,
                            predictorsPerTrial = 4,
                            minTimesEachVarUsed = 1,
                            logger = MonteCarloSimTestCase.logger)

        mcs.run()

        usedVars = set([n for t in mcs._trials for n in t.predictorNames])
        self.assertEqual(len(usedVars), len(mcs._allVars))
        
    # -------------------------------------------------------------------------
    # testRun
    # -------------------------------------------------------------------------
    def testRun(self):

        mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                            MonteCarloSimTestCase.outDir,
                            predictorsPerTrial = 4,
                            minTimesEachVarUsed = 0,
                            logger = MonteCarloSimTestCase.logger)
                 
        rf: RandomForestRegressor = mcs.run()
        
        # Minimum usage achieved.
        self.assertTrue(mcs.minVarUsageAchieved())

        # Non-zero permutation importance sufficient for top-N.
        averages: dict = mcs.computeAverages()
        
        self.assertLessEqual( \
            mcs.numVarsForFinalModel,
            sum(1 if a != 0 else 0 for a in averages.values()))
            
        # The top-N validity tests are satisfied before the final model run.
        self.assertEqual(len(rf.feature_names_in_), mcs.numVarsForFinalModel)
        
    # -------------------------------------------------------------------------
    # testRunTrials
    # -------------------------------------------------------------------------
    def testRunTrials(self):

        mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                            MonteCarloSimTestCase.outDir,
                            predictorsPerTrial = 4,
                            logger = MonteCarloSimTestCase.logger)
                            
        numCompleted = 0
        trials, numCompleted = mcs._runTrials(numCompleted)
        
    # -------------------------------------------------------------------------
    # testRunOneTrial
    # -------------------------------------------------------------------------
    def testRunOneTrial(self):

        mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                            MonteCarloSimTestCase.outDir,
                            predictorsPerTrial = 4,
                            logger = MonteCarloSimTestCase.logger)
                
        sampName = \
            mcs._masterTraining.dataset.schema.names[MasterTraining.SAMPLE_COL]
                        
        y: pd.DataFrame = \
             mcs._masterTraining.dataset.read([sampName]). \
             to_pandas().to_numpy().ravel()

        trial = mcs._runOneTrial()

    # -------------------------------------------------------------------------
    # testSaveFinalModel
    # -------------------------------------------------------------------------
    def testSaveFinalModel(self):
        
        mcs = MonteCarloSim(MonteCarloSimTestCase.trainingDir,
                            MonteCarloSimTestCase.outDir,
                            predictorsPerTrial = 4,
                            minTimesEachVarUsed = 0,
                            logger = MonteCarloSimTestCase.logger)
             
        rf1: RandomForestRegressor = mcs.run()
                       
        finalPath: Path = mcs.saveFinalModel(rf1)
        print('Model File:', finalPath)
        
        with open(finalPath, 'rb') as f:
            rf2: RandomForestClassifier = pickle.load(f)

    # -------------------------------------------------------------------------
    # debugPermutationImportances
    # -------------------------------------------------------------------------
    def debugPermutationImportances(self):

        trainingDir = Path('/explore/nobackup/projects/ilab/scratch/mcarrol2/'
                           'notebooks/vcf_clustering/new_parq')
                           
        mcs = MonteCarloSim(MonteCarloSimTestCase.base / 'training',
                            MonteCarloSimTestCase.base,
                            logger=MonteCarloSimTestCase.logger)
        
        trial: Trial = mcs._runOneTrial()
        print('PIs: ' + str(trial.permImportances))
