
from concurrent.futures import FIRST_EXCEPTION
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
import datetime as datetime
import glob
import joblib
import logging
import math
import multiprocessing
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from modis_vcf.model.MasterTraining import MasterTraining
from modis_vcf.model.TrainingType import TrainingType
from modis_vcf.model.Trial import Trial


# ----------------------------------------------------------------------------
# MonteCarloSim
#
# Use random forest from scikit-learn.
#
# You should be able to use default settings for the hyperparameters but
# please do take note of them.  I usually use
#
# Number of trees == 100
# Square root for the loss (or deciding) function
#
# And I forget what the other parameters are.  You can ask Caleb and Amanda 
# what they used with MODIS water for basic parameters if you need to.
#
# TODO: Validate input
# TODO: Important note: the output from the regressor should be "integer" as the input variables are also integers.  We do not need the output to be "float" because we don't care about the decimals and we don't have enough information in the input data to reliably predict decimals anyway. 
# ----------------------------------------------------------------------------
class MonteCarloSim(object):
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 trainingDir: Path,
                 trainingType: TrainingType, 
                 outDir: Path, 
                 predictorsPerTrial: int = 10, 
                 numVarsForFinalModel: int = 20,
                 minTimesEachVarUsed: int = 10,
                 procInputTrainFilesIndependently: bool = True,
                 numCpus: int = 1,
                 maxTrials: int = 20000,
                 logger: logging.RootLogger = None):
        
        if not logger:
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            if (not logger.hasHandlers()):

                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                logger.addHandler(ch)

        self._logger: logging.RootLogger = logger
        
        # Output directory
        if not outDir or not outDir.exists() or not outDir.is_dir():
            raise ValueError('A valid output directory must be provided.')
            
        self._outDir: Path = outDir
        
        self._trialDir: Path = outDir / 'trials'
        self._trialDir.mkdir(exist_ok=True)

        # Consolidate training-related initialization.
        self._trainingDir: Path = trainingDir
        self._trainingType: TrainingType = trainingType
        self._trials: list[Trial] = []
        self._masterTraining: MasterTraining = None
        self._allVars: list = None
        self._X: pd.DataFrame = None
        self._xTrain: pd.DataFrame = None
        self._xTest: pd.DataFrame = None
        self._yTrain: np.ndarray = None
        self._yTest: np.ndarray = None
        self._y: pd.DataFrame = None
            
        self._initTraining(procInputTrainFilesIndependently)

        # Initialize the simulation parameters.
        self._numVarsForFinalModel: int = numVarsForFinalModel or 20
        self._predictorsPerTrial: int = predictorsPerTrial or 10
        self._numCpus = min(numCpus, multiprocessing.cpu_count())
        self._maxTrials = maxTrials
        
        self._minTimesEachVarUsed: int = \
            minTimesEachVarUsed if minTimesEachVarUsed is not None else 10
        
        # Print configuration.
        logger.info('Training dir: ' + str(trainingDir))
        logger.info('Predictors per trial: ' + str(self._predictorsPerTrial))
        logger.info('Num rows: ' + str(self._masterTraining.numRows()))

        numVars = self._masterTraining.numVars()
        logger.info('Num vars: ' + str(numVars))
        
        if numVars < 263:
            
            warnings.warn('There are fewer available variables than '
                          'expected.')
        
        logger.info('Num vars for final model: ' + 
                    str(self._numVarsForFinalModel))
        
        logger.info('Min var usage: ' + str(self._minTimesEachVarUsed))
        logger.info('CPUs in use: ' + str(self._numCpus))
        logger.info('Max trials: ' + str(self._maxTrials))

    # ------------------------------------------------------------------------
    # initTraining
    #
    # Normally, all input training Parquet files are treated as a single
    # Parquet dataset.  The procInputTrainFilesIndependently option 
    # performs the test/train split on each training file in the training
    # directory, and combines all those into composite test/train data.
    # ------------------------------------------------------------------------
    def _initTraining(self, procTFilesIndependently: bool) -> None:
        
        self._masterTraining = MasterTraining(self._trainingDir,
                                              self._trainingType, 
                                              self._logger)

        self._allVars: list = self._masterTraining.dataset.schema.names \
                              [MasterTraining.START_COL:]
                              
        sampName = self._masterTraining.dataset.schema.names \
                   [MasterTraining.SAMPLE_COL]

        if procTFilesIndependently:

            xList = []
            yList = []
            xnList = []
            xtList = []
            ynList = []
            ytList = []
            
            for fName in self._masterTraining.dataset.files:
                
                x = ParquetDataset(fName).read().to_pandas()
                x = x.replace(-10001, 10001)
                y = x[sampName]
                yList.append(y)
                xList.append(x.drop(sampName, axis=1))

                xn, xt, yn, yt = train_test_split(x, y)
                xnList.append(xn)
                xtList.append(xt)
                ynList.append(yn)
                ytList.append(yt)
                
            self._X = pd.concat(xList)
            self._y = pd.concat(yList)
            self._xTrain = pd.concat(xnList)
            self._xTest = pd.concat(xtList)
            self._yTrain = pd.concat(ynList)
            self._yTest = pd.concat(ytList)

        else:

            X = self._masterTraining.dataset.read().to_pandas()
            self._X = X.replace(-10001, 10001)
            self._y = self._X[sampName]
            self._X.drop(sampName, axis=1, inplace=True)
            
            self._xTrain, self._xTest, self._yTrain, self._yTest = \
                train_test_split(self._X, self._y)

    # ------------------------------------------------------------------------
    # chooseColumns
    #
    # This one-liner is in its own method so the unit test can ensure the
    # seed is random.
    # ------------------------------------------------------------------------
    def _chooseColumns(self) -> list[str]:
        
        colNames = random.sample(self._allVars, self._predictorsPerTrial)
        return colNames
        
    # ------------------------------------------------------------------------
    # computeAverages
    # ------------------------------------------------------------------------
    def computeAverages(self) -> dict:
        
        # ---
        # Collate permutation importance for each variable.
        # permImports = {var1: [imp1, imp2, ...], var2: [imp1, imp2, ...]}
        # ---
        permImports = {k: [] for k in self._allVars}

        for trial in self._trials:
            
            impMeans = trial.permImportances['importances_mean']
        
            for i in range(len(impMeans)):

                varName = trial.predictorNames[i]
                permImports[varName].append(impMeans[i])
                
        # Compute the average for each 
        averages = dict.fromkeys(self._allVars, 0.0)
        
        for var in permImports:
            if permImports[var]:
                averages[var] = sum(permImports[var]) / len(permImports[var])
                
        return averages

    # ------------------------------------------------------------------------
    # getTopN
    # ------------------------------------------------------------------------
    def _getTopN(self, averages: dict) -> list:
        
        topN: list[Tuple] = \
            sorted(averages.items(), key=lambda x: x[1], reverse=True) \
            [:self._numVarsForFinalModel]

        topN = [tup[0] for tup in topN]
        
        # Ensure the top-N variables were actually used in trials.
        usedVars = set(np.array([t.predictorNames for t in self._trials]). \
            flatten().tolist())
            
        diff = set(topN).difference(usedVars)
        
        if len(diff) > 0:

            warnings.warn('Some top-n columns were not used in any trial.')
            return []
                               
        # ---
        # Ensure the top-N variables have an average permutation importance
        # different from zero.
        # ---
        for var in topN:
            
            if averages[var] == 0:
                
                warnings.warn('Top-n candidate variable, ' + 
                              var +
                              ' has an average permutation importance ' +
                              'of zero.')

                return []
            
        return topN
        
    # ------------------------------------------------------------------------
    # numTrials
    # ------------------------------------------------------------------------
    @property
    def numTrials(self) -> int:
        return self._numTrials
        
    # ------------------------------------------------------------------------
    # numVarsForFinalModel
    # ------------------------------------------------------------------------
    @property
    def numVarsForFinalModel(self) -> int:
        return self._numVarsForFinalModel
        
    # ------------------------------------------------------------------------
    # minVarUsageAchieved
    # ------------------------------------------------------------------------
    def minVarUsageAchieved(self) -> bool:
        
        varUsageCount = dict.fromkeys(self._allVars, 0)

        # Count variable usage in all trials.
        for trial in self._trials:
            
            for varName in trial.predictorNames:
                
                varUsageCount[varName] += 1
            
        # Test usage achievement.
        for var in varUsageCount:
            
            if varUsageCount[var] < self._minTimesEachVarUsed:
                return False
            
        return True

    # ------------------------------------------------------------------------
    # printSortedPredictors
    # ------------------------------------------------------------------------
    def _printSortedPredictors(self, averages: dict) -> list:
        
        sortedPreds: list[Tuple] = \
            sorted(averages.items(), key=lambda x: x[1], reverse=True)
            
        sortedPreds = [x[0] for x in sortedPreds]
        
        timeStamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        
        outName = 'SortedMetrics-' + \
                  self._trainingType.value + \
                  timeStamp + \
                  '.txt'
                  
        outPath: Path = self._outDir / outName
        
        with open(outPath, 'w') as f:
           f.write('\n'.join(str(i) for i in sortedPreds)) 

    # ------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------
    def run(self) -> RandomForestRegressor:

        topN = []
        numCompleted = 0
        varUsageCount = dict.fromkeys(self._allVars, 0)
        rf: RandomForestRegressor = None
        allConditionsMet = False
        averages = {}
        
        while not allConditionsMet:
            
            # This is mostly for testing, so test complete quickly.
            if numCompleted >= self._maxTrials:
                
                self._logger.warn('Insufficient trials were run because ' + 
                                  'the maximum trials is too low. ' +
                                  'Selecting a bogus set of variables ' +
                                  'for the final model.')
                                  
                allConditionsMet = True
                topN = self._allVars[0:self._numVarsForFinalModel]
                break
                
            # ---
            # Run a batch of trials.  The batch size depends on the number
            # of total trials requested and the number of CPUs.
            # ---
            batchOfTrials, numCompleted = self._runTrials(numCompleted)
            self._trials += batchOfTrials
            
            # Monitor variable usage.
            if not self.minVarUsageAchieved():
            
                self._logger.info('Variable usage unsatisfied.')
                continue

            # Compute average permutation importance.
            averages = self.computeAverages()
            
            # Monitor non-zero permutation importance.
            if sum(1 if a != 0 else 0 for a in averages.values()) < \
                self._numVarsForFinalModel:

                self._logger.info('Not enough non-zero permutation '
                                  'importances to satisfy the number of '
                                  'variables required for the final model.')
                
                continue
            
            # ---
            # Compute top-n predictors.  The _getTopN() method monitors itself,
            # returning an empty list when its conditions are unsatisfied.
            # This is inconsistent with this method, run(), because the other
            # conditions are monitored here.
            # ---
            topN = self._getTopN(averages)

            # Monitor top-n.
            if len(topN):
            
                topnMet = True
            
            else:
                
                if not topnMet:
                    self._logger.info('Top-n unsatified.')
                    
            allConditionsMet = True
                        
        # Run the final model.
        if allConditionsMet:
            
            rf = self._runRandomForest(self._xTrain[topN], self._yTrain)
            self._printSortedPredictors(averages)
            
        return rf
        
    # ------------------------------------------------------------------------
    # runTrials
    # ------------------------------------------------------------------------
    def _runTrials(self, numCompleted: int) -> (list, int):
        
        with ProcessPoolExecutor(max_workers=self._numCpus) as xtor:

            completedTrials = []

            self._logger.info('Starting a batch of ' + 
                              str(self._numCpus) + 
                              ' trials.')

            futures = []

            for count in range(self._numCpus):
                futures.append(xtor.submit(self._runOneTrial))

            self._logger.info('Awaiting batch completion.')
            
            comp, incomp = wait(futures, return_when=FIRST_EXCEPTION)
            numCompleted += len(comp)
            
            for future in comp:
                
                trial: Trial = future.result()
                completedTrials.append(trial)
                
        self._logger.info('Total completed: ' + str(numCompleted))
        
        return completedTrials, numCompleted

    # ------------------------------------------------------------------------
    # runOneTrial
    #
    # Need all rows for each of predictorsPerTrial columns, plus one more 
    # column, the training column.
    # ------------------------------------------------------------------------
    def _runOneTrial(self) -> Trial:
        
        colNames: list[str] = self._chooseColumns()
        X = self._X[colNames]
        xTrain = self._xTrain[colNames]

        rf: RandomForestRegressor = \
            self._runRandomForest(xTrain, self._yTrain)

        # ---
        # According to the user manual, permutation importance items are
        # presented in the same order as the input variables.  Wish this were
        # explicit.  For the trial object, the permutation importance values
        # correspond, in order, to the predictor names.  See
        # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
        #
        # The permutation importances are all 0.  RFR documentation suggests
        # using permutation_importance(), so this implies it is compatible.
        # ---
        # permImportances = dict(permutation_importance(rf, X, self._y))
        permImportances = permutation_importance(rf, X, self._y)
        name = 'Trial-' + str(hash(''.join(colNames)))
        trial = Trial(name, colNames, permImportances)
        trial.save(self._trialDir)

        return trial
        
    # ------------------------------------------------------------------------
    # runRandomForest
    #
    # This is a template method.
    # ------------------------------------------------------------------------
    def _runRandomForest(self, xTrain, yTrain) -> RandomForestRegressor:

        rf = RandomForestRegressor(n_estimators=1)
        rf = rf.fit(xTrain, yTrain)
        return rf

    # ------------------------------------------------------------------------
    # saveFinalModel
    # ------------------------------------------------------------------------
    def saveFinalModel(self, finalModel: RandomForestRegressor) -> Path:
        
        outPath: Path = self._outDir / (self._trainingType.value + '.bin')
        
        with open(outPath, 'wb') as f:
            joblib.dump(finalModel, f)

        return outPath
                