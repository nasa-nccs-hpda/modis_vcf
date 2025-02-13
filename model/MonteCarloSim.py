
from concurrent.futures import FIRST_EXCEPTION
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
import glob
import joblib
import logging
import multiprocessing
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from modis_vcf.model.MasterTraining import MasterTraining
from modis_vcf.model.Trial import Trial

DEFAULT_VALUE = 10
DEFAULT_TOP_N = 20
    

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
# TODO: Make the process flow more direct and easier to follow.
#       saveFinalModel -> finalModel -> topN -> averages -> self.trials ->
#       trials -> runTrials
# TODO:  Important note: the output from the regressor should be "integer" as the input variables are also integers.  We do not need the output to be "float" because we don't care about the decimals and we don't have enough information in the input data to reliably predict decimals anyway. 
# ----------------------------------------------------------------------------
class MonteCarloSim(object):
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 trainingDir: Path, 
                 numTrials: int = DEFAULT_VALUE, 
                 predictorsPerTrial: int = DEFAULT_VALUE, 
                 numVarsForFinalModel: int = DEFAULT_TOP_N,
                 minTimesEachVarUsed: int = DEFAULT_VALUE,
                 numCpus: int = 1,
                 logger: logging.RootLogger = None):
        
        if not logger:
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            if (not logger.hasHandlers()):

                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                logger.addHandler(ch)

        self._logger: logging.RootLogger = logger

        # None could be passed, overriding the default value, so ...
        self._numTrials: int = numTrials or DEFAULT_VALUE
        self._predictorsPerTrial: int = predictorsPerTrial or DEFAULT_VALUE
        
        # This is only for an experiment with MonteCarloSimGpu.
        self._trainingDir: Path = trainingDir
        
        self._masterTraining = MasterTraining(trainingDir, logger)
        
        self._allVars: list = self.masterTraining.dataset.schema.names \
                              [MasterTraining.START_COL:]

        self._numCpus: int = numCpus or 1
        
        # ---
        # This is the number of highest-performing variables to select for
        # the final model.  In other words, the top ten (top
        # numVarsForFinalModel).
        # ---
        self._numVarsForFinalModel: int = numVarsForFinalModel or DEFAULT_TOP_N
        
        # ---
        # Each variable must be randomly selected at least minTimesEachVarUsed
        # times before the accumulation of trials may stop.  This overrides
        # self._numTrials, if self._numTrials is reached before
        # minTimesEachVarUsed is satisfied.
        # ---
        self._minTimesEachVarUsed: int = \
            minTimesEachVarUsed if minTimesEachVarUsed is not None else \
            DEFAULT_VALUE
        
        # ---
        # Prepare y.
        # ---
        sampName = self.masterTraining.dataset.schema.names \
                   [MasterTraining.SAMPLE_COL]
                        
        y: pd.DataFrame = \
             self.masterTraining.dataset.read([sampName]). \
             to_pandas().to_numpy().ravel()

        self._y = np.where(y == -10001, 10001, y)

        # Print configuration.
        logger.info('Training dir: ' + str(trainingDir))
        logger.info('Num trials: ' + str(self._numTrials))
        logger.info('Predictors per trial: ' + str(self._predictorsPerTrial))
        logger.info('Num rows: ' + str(self._masterTraining.numRows()))

        numVars = self._masterTraining.numVars()
        logger.info('Num vars: ' + str(numVars))
        
        if numVars < 263:
            logger.warn('There are fewer available variables than expected.')
        
        logger.info('Num vars for final model: ' + 
                    str(self._numVarsForFinalModel))
        
        logger.info('Min var usage: ' + str(self._minTimesEachVarUsed))
        logger.info('CPUs in use: ' + str(self._numCpus))

    # ------------------------------------------------------------------------
    # allVars
    # ------------------------------------------------------------------------
    @property
    def allVars(self) -> list:
        return self._allVars
        
    # ------------------------------------------------------------------------
    # chooseColumns
    #
    # This one-liner is in its own method so the unit test can ensure the
    # seed is random.
    # ------------------------------------------------------------------------
    def _chooseColumns(self) -> list[str]:
        
        colNames = random.sample(self.allVars, self.predictorsPerTrial)
        return colNames
        
    # ------------------------------------------------------------------------
    # computeAverages
    # ------------------------------------------------------------------------
    def _computeAverages(self, trials) -> dict:
        
        # ---
        # Collate permutation importance for each variable.
        # permImports = {var1: [imp1, imp2, ...], var2: [imp1, imp2, ...]}
        # ---
        permImports = {k: [] for k in self.allVars}

        for trial in trials:
            
            impMeans = trial.permImportances['importances_mean']
        
            for i in range(len(impMeans)):

                varName = trial.predictorNames[i]
                permImports[varName].append(impMeans[i])
                
        # Compute the average for each 
        averages = dict.fromkeys(self.allVars, 0.0)
        
        for var in permImports:
            
            if permImports[var]:
                
                sumI = sum(permImports[var])
                averages[var] = sumI / len(permImports[var])
                
        return averages

    # ------------------------------------------------------------------------
    # masterTraining
    # ------------------------------------------------------------------------
    @property
    def masterTraining(self) -> MasterTraining:
        return self._masterTraining
        
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
    # pollVarUsage
    # ------------------------------------------------------------------------
    def _pollVarUsage(self, varUsageCount: dict) -> bool:
        
        usageAchieved = True

        if self._minTimesEachVarUsed == 0:
            return usageAchieved
            
        for var in varUsageCount:

            if varUsageCount[var] < self._minTimesEachVarUsed:

                usageAchieved = False
                break
                
        return usageAchieved
            
    # ------------------------------------------------------------------------
    # predictorsPerTrial
    # ------------------------------------------------------------------------
    @property
    def predictorsPerTrial(self) -> int:
        return self._predictorsPerTrial
        
    # ------------------------------------------------------------------------
    # prepX
    # ------------------------------------------------------------------------
    def _prepX(self, colNames: list) -> \
        [pd.core.frame.DataFrame, 
         pd.core.frame.DataFrame, 
         pd.core.frame.DataFrame, 
         np.ndarray, 
         np.ndarray]:
        
        # Read the columns.  Sklearn cannot use Pyarrow.Table.
        X: pd.DataFrame = \
            self.masterTraining.dataset.read(colNames).to_pandas()
        
        # No-data value, -10001, must be changed to +10001.
        X = X.where(X == -10001, 10001)
        
        # Split the columns into test and training subsets.
        xTrain, xTest, yTrain, yTest = train_test_split(X, self._y)
        
        return X, xTrain, xTest, yTrain, yTest
        
    # ------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------
    def run(self) -> RandomForestRegressor:

        trials: list[Trial] = self._runTrials()
        averages: dict = self._computeAverages(trials)
        
        topN: list[Tuple] = \
            sorted(averages.items(), key=lambda x: x[1], reverse=True) \
            [:self._numVarsForFinalModel]

        topNCols: list = [tup[0] for tup in topN]

        X, xTrain, xTest, yTrain, yTest = self._prepX(topNCols)
        rf: RandomForestRegressor = self._runRandomForest(xTrain, yTrain)

        return rf
        
    # ------------------------------------------------------------------------
    # runTrials
    # ------------------------------------------------------------------------
    def _runTrials(self) -> list:
        
        with ProcessPoolExecutor(max_workers=self._numCpus) as xtor:

            # Tally variable usage for self._minTimesEachVarUsed.
            varUsageCount = dict.fromkeys(self.allVars, 0) 
            usageAchieved = False
            completedTrials = []
            completed = 0
        
            while not usageAchieved:

                self._logger.info('Starting a batch of ' + 
                                  str(self._numCpus) + 
                                  ' trials.')

                futures = []

                for count in range(self._numCpus):
                    futures.append(xtor.submit(self._runOneTrial))

                self._logger.info('Awaiting batch completion.')
                
                comp, incomp = wait(futures, return_when=FIRST_EXCEPTION)
                completed += len(comp)
                
                self._logger.info('Complete: ' + str(len(comp)))
                self._logger.info('Incomplete: ' + str(len(incomp)))
                self._logger.info('Total completed: ' + str(completed))

                for future in comp:
                    
                    trial: Trial = future.result()
                    completedTrials.append(trial)
                    
                    # Tally variable usage.
                    for var in trial.predictorNames:
                        varUsageCount[var] += 1
            
                # Do not bother polling usage until miniumum trials satisfied.
                if completed >= self._numTrials:
            
                    usageAchieved = self._pollVarUsage(varUsageCount)
                    
                    self._logger.info('Minimum variable usage achieved: ' +
                                      str(usageAchieved))
                
        self._logger.info('Trials complete.')
        self._logger.info('Num trials: ' + str(len(completedTrials)))
        
        return completedTrials

    # ------------------------------------------------------------------------
    # runOneTrial
    #
    # Need all rows for each of predictorsPerTrial columns, plus one more 
    # column, the training column.
    # ------------------------------------------------------------------------
    def _runOneTrial(self) -> Trial:
        
        colNames: list[str] = self._chooseColumns()
        X, xTrain, xTest, yTrain, yTest = self._prepX(colNames)

        rf: RandomForestRegressor = self._runRandomForest(xTrain, yTrain)

        # ---
        # According to the user manual, permutation importance items are
        # presented in the same order as the input variables.  Wish this were
        # explicit.  For the trial object, the permutation importance values
        # correspond, in order, to the predictor names.  See
        # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
        # ---
        permImportances = dict(permutation_importance(rf, X, self._y))
        name = 'Trial-' + str(hash(''.join(colNames)))
        trial = Trial(name, colNames, permImportances)

        return trial
        
    # ------------------------------------------------------------------------
    # runRandomForest
    #
    # RFR can use n_jobs to send each fit() to a different CPU.  We use a
    # unique RFR for each trial, so n_jobs does not help.
    # ------------------------------------------------------------------------
    def _runRandomForest(self, xTrain, yTrain) -> RandomForestRegressor:

        rf = RandomForestRegressor(n_estimators=1)
        rf = rf.fit(xTrain, yTrain)
        return rf

    # ------------------------------------------------------------------------
    # saveFinalModel
    # ------------------------------------------------------------------------
    def saveFinalModel(self, 
                       outDir: Path, 
                       finalModel: RandomForestRegressor) -> Path:
        
        outPath: Path = outDir / ('MCS-model.bin')
        
        with open(outPath, 'wb') as f:
            joblib.dump(finalModel, f)

        return outPath
                