
import glob
import logging
from pathlib import Path
import random

import pyarrow as pa
from pyarrow.parquet import ParquetDataset

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
                 logger: logging.RootLogger = None):
        
        self._numTrials: int = numTrials
        self._predictorsPerTrial: int = predictorsPerTrial
        self._masterTraining = MasterTraining(trainingDir, logger)
        
    # ------------------------------------------------------------------------
    # trainingDs
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
    # predictorsPerTrial
    # ------------------------------------------------------------------------
    @property
    def predictorsPerTrial(self) -> int:
        return self._predictorsPerTrial
        
    # ------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------
    def run(self):
        
        trials = []
        
        for trialNum in range(1, self._numTrials + 1):

            trial: Trial = self._runOneTrial(trialNum)
            trials.append(trial)
            
    # ------------------------------------------------------------------------
    # runOneTrial
    #
    # Need all rows for each of predictorsPerTrial columns, plus one more 
    # column, the training column.
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    #
    # predictions = rf.predict(xTest)
    # acc = accuracy_score(yTest, predictions)
    # ------------------------------------------------------------------------
    def _runOneTrial(self, trialNum: int) -> Trial:
        
        name = 'Trial-' + str(trialNum)
        
        # Randomly choose among the columns, omitting the index-related ones.
        allCols = self.masterTraining.dataset.schema.names \
                  [MasterTraining.START_COL:]

        colNames = random.sample(allCols, self.predictorsPerTrial)
        
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

        permImportance = dict(permutation_importance(rf, X, y))
        trial = Trial(name, colNames, permImportance)
        
        return trial
        