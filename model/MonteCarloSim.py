
import glob
import logging
from pathlib import Path
import random

from pyarrow.parquet import ParquetDataset

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
    def __init__(self, trainingDir: Path, logger: logging.RootLogger):
        
        self._numTrials: int = 10
        self._predictorsPerTrial: int = 10
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
    # randomizedSelection
    #
    # If I use and one-dimensional array, I can use random.choices().
    # ------------------------------------------------------------------------
    # def _randomizedSelection(self) -> list:
    #
    #     trials = []
    #
    #     # ---
    #     # Make a flat array of length rows x cols.  Select in 1D, then
    #     # translate to 2D later.
    #     # ---
    #     flat = range(self._masterTraining.numRows * self._masterTraining.numCols)
    #
    #     for trial in range(1, self._numTrials + 1):
    #
    #         name = 'Trial-' + str(trial)
    #         pts1D = random.choices(flat, k = self._predictorsPerTrial)
    #
    #         sampleLocs = \
    #             [(int(p / self._masterTraining.numCols), \
    #              p % self._masterTraining.numCols) for p in pts1D]
    #
    #         trials.append(Trial(name, sampleLocs))
    #
    #     return trials

    # ------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------
    def run(self):
        
        for trial in range(1, self._numTrials + 1):

            trial: Trial = self._runOneTrial(trial)
            
    # ------------------------------------------------------------------------
    # runOneTrial
    #
    # Need all rows for each of predictorsPerTrial columns, plus one more 
    # column, the training column.
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # ------------------------------------------------------------------------
    def _runOneTrial(self, trialNum: int) -> Trial:
        
        name = 'Trial-' + str(trial)
        
        # Randomly choose among the columns.
        cols = random.choices(range(self.masterTraining.numCols))
        
        