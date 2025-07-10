
import gc
from pathlib import Path
import logging

import numpy as np
import pandas as pd

import cudf
from cudf.core.dataframe import DataFrame as cudfDataFrame
from cuml.ensemble import RandomForestRegressor as cumlRF_Regressor
from cuml.metrics import r2_score
import cupy as cp

from modis_vcf.model.MonteCarloSim import MonteCarloSim
from modis_vcf.model.TrainingType import TrainingType
from modis_vcf.model.Trial import Trial


# ----------------------------------------------------------------------------
# MonteCarloSimGpu
#
# "There is no smart ProcessPoolExecutor for GPUs, I would probably just create a pool of jobs, and use the CUDA_VISIBLE_DEVICES variable to assign the job to a specific GPU that is not busy."
#
# echo $CUDA_VISIBLE_DEVICES ==> 0,1,2,3
# nvidia-smi
# import pynvml
# ----------------------------------------------------------------------------
class MonteCarloSimGpu(MonteCarloSim):
    
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


        super(MonteCarloSimGpu, self).__init__(
            trainingDir,
            trainingType,
            outDir,
            predictorsPerTrial,
            numVarsForFinalModel,
            minTimesEachVarUsed,
            procInputTrainFilesIndependently,
            numCpus,
            maxTrials,
            logger)
            
    # ------------------------------------------------------------------------
    # permutationImportance
    # ------------------------------------------------------------------------
    def _permutationImportance(self, 
                               rf: cumlRF_Regressor, 
                               colNames: list[str]) -> []:
        
        xTest: cudfDataFrame = \
            cudf.read_parquet(self._xTestPath,columns=colNames)

        xTest = xTest.astype(np.float32)
        
        # The iloc part converts the DataFrame to a Series.
        yTest = cudf.read_parquet(self._yTestPath).iloc[:, 0]

        pred: cudf.core.series.Series = rf.predict(xTest)
        baseLineScore: float = r2_score(yTest, pred)
        
        cp.random.seed(42)
        numRepeats = 5
        importances = cp.zeros((xTest.shape[1], numRepeats), dtype=cp.float32)
        
        featureNames = list(xTest.columns)

        for i, col in enumerate(featureNames):

            for j in range(numRepeats):

                xtPermuted = xTest.copy(deep=True)
                colValue = xtPermuted[col].to_cupy(copy=True)
                cp.random.shuffle(colValue)
                xtPermuted[col] = colValue
                
                pred = rf.predict(xtPermuted)
                permutedScore = r2_score(yTest, pred)
                importances[i, j] = baseLineScore - permutedScore

        importancesMean: np.ndarray = importances.mean(axis=1).get()
        importancesStd: np.ndarray = importances.std(axis=1).get()
        
        pi = {'importances_mean' : importancesMean,
              'importances_std' : importancesStd,
              'importances' : importances.get()}
              
        return pi
        
    # ------------------------------------------------------------------------
    # runTrials
    # ------------------------------------------------------------------------
    def _runTrials(self, numCompleted: int) -> (list, int):

        # Run just one, at least for now.
        self._logger.info('Starting a batch of 1 trial.')
        trial: Trial = self._runOneTrial()
        numCompleted += 1
        return ([trial], numCompleted)
        
    # ------------------------------------------------------------------------
    # runOneTrial
    # ------------------------------------------------------------------------
    def _runOneTrial(self) -> Trial:

        colNames: list[str] = self._chooseColumns()
        
        xTrain: cudfDataFrame = \
            cudf.read_parquet(self._xTrainPath, columns=colNames)

        xTrain = xTrain.astype(np.float32)
        yTrain = cudf.read_parquet(self._yTrainPath).to_numpy().ravel()
        
        rf: cumlRF_Regressor = self._runRandomForest(xTrain, yTrain)

        del xTrain
        del yTrain
        gc.collect()
        
        self._logger.info('Fit complete.  Computing permutation importances.')
        permImportances = self._permutationImportance(rf, colNames)

        name = 'Trial-' + str(hash(''.join(colNames)))
        trial = Trial(name, colNames, permImportances)
        trial.save(self._trialDir)

        return trial

    # ------------------------------------------------------------------------
    # runRandomForest
    # ------------------------------------------------------------------------
    def _runRandomForest(self,
                         xTrain: cudfDataFrame,
                         yTrain: np.ndarray) -> cumlRF_Regressor:

        rf: cumlRF_Regressor = cumlRF_Regressor(n_estimators=1,
                                                max_depth=16,
                                                max_features=1.0,
                                                n_streams=4,
                                                random_state=42)

        rf = rf.fit(xTrain, yTrain)

        return rf
