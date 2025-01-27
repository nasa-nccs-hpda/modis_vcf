
import glob
import pickle
import logging
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import pyarrow as pa

from sklearn.ensemble import RandomForestClassifier
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
        self._masterTraining = MasterTraining(trainingDir, logger)
        
        self._allVars: list = self.masterTraining.dataset.schema.names \
                              [MasterTraining.START_COL:]

        # ---
        # Save certain intermediate variables that might be helpful to 
        # interrogate later.
        # ---
        self._averages: dict = None
        self._topN: list[tuple] = None
        self._trials: list[Trial] = None
        
        # ---
        # This is the number of highest-performing variables to select for
        # the final model.  In other words, the top ten (top numVarsForFinalModel).
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
        
        logger.info('Num vars for final model: ' + 
                    str(self._numVarsForFinalModel))
        
        logger.info('Min var usage: ' + str(self._minTimesEachVarUsed))

    # ------------------------------------------------------------------------
    # allVars
    # ------------------------------------------------------------------------
    @property
    def allVars(self) -> list:
        return self._allVars
        
    # ------------------------------------------------------------------------
    # averages
    # ------------------------------------------------------------------------
    @property
    def averages(self) -> dict:
        
        if not self._averages:
            self._averages = self._computeAverages(self.trials)
            
        return self._averages
        
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
    # _finalModel
    # ------------------------------------------------------------------------
    def _finalModel(self) -> RandomForestClassifier:
        
        topPredictors = [t[0] for t in self.topN]
        X, xTrain, xTest, yTrain, yTest = self._prepX(topPredictors)
        rf: RandomForestClassifier = self._runRandomForest(xTrain, yTrain)
        return rf
        
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
    #
    # TODO: convert to cudf.DataFrame, instead of pd.DataFrame
    # ------------------------------------------------------------------------
    def _prepX(self, colNames: list) -> \
        [pd.core.frame.DataFrame, 
         pd.core.frame.DataFrame, 
         np.ndarray, 
         np.ndarray]:
        
        # Read the columns.  Sklearn cannot use Pyarrow.Table.
        X: pd.DataFrame = \
            self.masterTraining.dataset.read(colNames).to_pandas()
        
        # No-data value, -10001, must be changed to +10001.
        X = X.where(X != -10001, 10001)
        
        # Split the columns into test and training subsets.
        xTrain, xTest, yTrain, yTest = train_test_split(X, self._y)
        
        return X, xTrain, xTest, yTrain, yTest
        
    # ------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------
    def run(self) -> RandomForestClassifier:

        rf: RandomForestClassifier = self._finalModel()
        return rf
        
    # ------------------------------------------------------------------------
    # runTrials
    # ------------------------------------------------------------------------
    def _runTrials(self) -> list:
        
        self._logger.info('Starting trials.')

        # Tally variable usage for self._minTimesEachVarUsed.
        varUsageCount = dict.fromkeys(self.allVars, 0) 
        usageAchieved = False
        trials = []
        trialNum = 0
        
        while not usageAchieved:

            trialNum += 1
            
            self._logger.info('Running trial ' + 
                              str(trialNum) + 
                              ' of ' + 
                              str(self._numTrials))
                              
            trial: Trial = self._runOneTrial(trialNum)
            trials.append(trial)
            
            # Tally variable usage.
            for var in trial.predictorNames:
                varUsageCount[var] += 1
                
            # Do not bother polling usage until miniumum trials satisfied.
            if trialNum >= self._numTrials:
                
                self._logger.info('Minimum trials achieved.  ' + \
                                  'Checking minimum variable usage.')
                      
                usageAchieved = self._pollVarUsage(varUsageCount)  
                
        self._logger.info('Trials completed.') 
        
        return trials

    # ------------------------------------------------------------------------
    # runOneTrial
    #
    # Need all rows for each of predictorsPerTrial columns, plus one more 
    # column, the training column.
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # ------------------------------------------------------------------------
    def _runOneTrial(self, trialNum: int) -> Trial:
        
        name = 'Trial-' + str(trialNum)
        colNames: list[str]  = self._chooseColumns()
        X, xTrain, xTest, yTrain, yTest = self._prepX(colNames)
        rf: RandomForestClassifier = self._runRandomForest(xTrain, yTrain)

        # ---
        # According to the user manual, permutation importance items are
        # presented in the same order as the input variables.  Wish this were
        # explicit.  For the trial object, the permutation importance values
        # correspond, in order, to the predictor names.  See
        # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
        # ---
        permImportances = dict(permutation_importance(rf, X, self._y))
        trial = Trial(name, colNames, permImportances)

        return trial
        
    # ------------------------------------------------------------------------
    # runRandomForest
    # ------------------------------------------------------------------------
    def _runRandomForest(self, xTrain, yTrain) -> RandomForestClassifier:

        rf = RandomForestClassifier(n_estimators=1)  # single decision tree
        rf = rf.fit(xTrain, yTrain)
        return rf

    # ------------------------------------------------------------------------
    # saveFinalModel
    # ------------------------------------------------------------------------
    def saveFinalModel(self, outDir: Path) -> Path:
        
        finalModel = self._finalModel()
        outBase: str = Path(self.masterTraining.dataset.files[0]).name
        outName: Path = Path(outBase).stem
        
        outPath: Path = outDir / (outName + '-model.bin')
        
        with open(outPath, 'wb') as f:
            pickle.dump(finalModel, f)

        return outPath
        
    # ------------------------------------------------------------------------
    # topN
    # ------------------------------------------------------------------------
    @property
    def topN(self) -> list[tuple]:
        
        if not self._topN:
            
            self._topN = sorted(self.averages.items(),
                                key=lambda x: x[1],
                                reverse=True)[:self._numVarsForFinalModel]
        return self._topN
        
    # ------------------------------------------------------------------------
    # trials
    # ------------------------------------------------------------------------
    @property
    def trials(self) -> list:
        
        if not self._trials:
            self._trials = self._runTrials()
            
        return self._trials
        