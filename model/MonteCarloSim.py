
import glob
import logging
from pathlib import Path
import random

import pyarrow as pa

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from modis_vcf.model.MasterTraining import MasterTraining
from modis_vcf.model.Trial import Trial


# ----------------------------------------------------------------------------
# MonteCarloSim
#
# Use random forest from scikit-learn.
#
# You should be able to use default settings for the hyperparameters but please
# do take note of them.  I usually use
#
# Number of trees == 100
# Square root for the loss (or deciding) function
#
# And I forget what the other parameters are.  You can ask Caleb and Amanda 
# what they used with MODIS water for basic parameters if you need to.
# ----------------------------------------------------------------------------
class MonteCarloSim(object):
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 trainingDir: Path, 
                 numTrials: int = 10, 
                 predictorsPerTrial: int = 10, 
                 numVarsToSelect: int = 10,
                 minTimesEachVarUsed: int = 10,
                 logger: logging.RootLogger = None):
        
        self._numTrials: int = numTrials
        self._predictorsPerTrial: int = predictorsPerTrial
        self._logger = logger
        self._masterTraining = MasterTraining(trainingDir, logger)
        
        self._allVars = self.masterTraining.dataset.schema.names \
                        [MasterTraining.START_COL:]

        # ---
        # This is the number of highest-performing variables to select for
        # the final model.  In other words, the top ten (top numVarsToSelect).
        # ---
        self._numVarsToSelect: int = numVarsToSelect
        
        # ---
        # Each variable must be randomly selected at least minTimesEachVarUsed
        # times before the accumulation of trials may stop.  This overrides
        # self._numTrials, if self._numTrials is reached before
        # minTimesEachVarUsed is satisfied.
        # ---
        self._minTimesEachVarUsed: int = minTimesEachVarUsed

    # ------------------------------------------------------------------------
    # allVars
    # ------------------------------------------------------------------------
    @property
    def allVars(self) -> list:
        return self._allVars
        
    # ------------------------------------------------------------------------
    # rankVars
    # ------------------------------------------------------------------------
    def rankVars(self, trials) -> dict:
        
        # Collate permutation importance for each variable.
        collatedImportance = dict.fromkeys(self.allVars, float)

        for trial in trials:
            
            means = trial.permImportance['importances_mean']
        
            for i in range(len(means)):

                varName = trial.predictorNames[i]
                varMean = means[i]
                curMean = collatedImportance[varName]
                collatedImportance[varName] = (curMean + varMean) / 2.0
                
        return collatedImportance
        
    # ------------------------------------------------------------------------
    # numTrials
    # ------------------------------------------------------------------------
    @property
    def numTrials(self) -> int:
        return self._numTrials
        
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
    # run
    # ------------------------------------------------------------------------
    def run(self):
        
        trials: list = self._runTrials()
        rankedVars: dict = self.rankVars(trials)
        
    # ------------------------------------------------------------------------
    # runTrials
    # ------------------------------------------------------------------------
    def _runTrials(self) -> list[Trial]:
        
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
        
        # Randomly choose among the columns, omitting the index-related ones.
        colNames = random.sample(self.allVars, self.predictorsPerTrial)
        
        # Read the columns.  Sklearn cannot use Pyarrow.Table.
        X: pd.DataFrame = \
            self.masterTraining.dataset.read(colNames).to_pandas()
        
        sampName = self.masterTraining.dataset.schema.names \
                        [MasterTraining.SAMPLE_COL]
                        
        y: pd.DataFrame = \
             self.masterTraining.dataset.read([sampName]). \
             to_pandas().to_numpy().ravel()
        
        # Split the columns into test and training subsets.
        xTrain, xTest, yTrain, yTest = train_test_split(X, y)
        
        # Fit the model.
        rf = RandomForestClassifier()
        rf = rf.fit(xTrain, yTrain)

        # ---
        # According to the user manual, permutation importance items are
        # presented in the same order as the input variables.  Wish this were
        # explicit.  For the trial object, the permutation importance values
        # correspond, in order, to the predictor names.  See
        # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
        # ---
        permImportance = dict(permutation_importance(rf, X, y))
        trial = Trial(name, colNames, permImportance)
        
        return trial

    # ------------------------------------------------------------------------
    # trainingDs
    # ------------------------------------------------------------------------
    @property
    def masterTraining(self) -> MasterTraining:
        return self._masterTraining
        