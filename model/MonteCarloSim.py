
from concurrent.futures import FIRST_EXCEPTION
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
import datetime as datetime
import gc
import glob
import joblib
import logging
import math
import multiprocessing
from pathlib import Path
import pickle
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
        
        # Directories
        if not outDir or not outDir.exists() or not outDir.is_dir():
            raise ValueError('A valid output directory must be provided.')
            
        self._outDir: Path = outDir
        self._trialDir: Path = outDir / 'trials'
        self._trialDir.mkdir(exist_ok=True)
        self._trainingDir: Path = trainingDir

        # Labeled training data
        self._trainingType: TrainingType = trainingType

        masterTraining = MasterTraining(self._trainingDir,
                                        self._trainingType, 
                                        self._logger)

        self._allVars: list = masterTraining.dataset.schema.names \
                              [MasterTraining.START_COL:]

        # ---
        # Load X- and y-related data members.  The test values are needed
        # for the GPU version's permutation importance computation.  X
        # objects are large, so save them as Parquet files.  Pertinent columns
        # are extracted as needed, minimizing memory usage.  We must write y;
        # otherwise we would need to read it all again just to get y.
        # ---
        self._xPath = self._outDir / ('X.parq')
        self._xTrainPath: Path = self._outDir / ('xTrain.parq')
        self._xTestPath: Path = self._outDir / ('xTest.parq')
        
        self._yPath: Path = self._outDir / ('y.parq')
        self._yTrainPath: Path = self._outDir / ('yTrain.parq')
        self._yTestPath: Path = self._outDir / ('yTest.parq')
        
        self._initXY(masterTraining, procInputTrainFilesIndependently)

        # Initialize the simulation parameters.
        tFiles: list = self._trialDir.glob('Trial-*.bin')
        self._trials: list[Trial] = [Trial.load(t) for t in tFiles]
        self._numTrials = len(self._trials)

        self._numVarsForFinalModel: int = numVarsForFinalModel or 20
        self._predictorsPerTrial: int = predictorsPerTrial or 10
        self._numCpus = min(numCpus, multiprocessing.cpu_count())
        self._maxTrials = maxTrials
        
        self._minTimesEachVarUsed: int = \
            minTimesEachVarUsed if minTimesEachVarUsed is not None else 10
        
        # ---
        # The fit model.  CUML's regressor does not use feature names.  These
        # are needed when we make prediction because it determines which
        # metrics must be created.  Store the top-N predictors here, so they
        # may be written to disk along with the final model.
        # ---
        self._rf: RandomForestRegressor = None
        self._topN: list = None
        
        # Print configuration.
        logger.info('Training dir: ' + str(trainingDir))
        logger.info('Predictors per trial: ' + str(self._predictorsPerTrial))
        logger.info('Num rows: ' + str(masterTraining.numRows()))

        numVars = masterTraining.numVars()
        logger.info('Num vars: ' + str(numVars))
        
        if numVars < 263:
            
            warnings.warn('There are fewer available variables than '
                          'expected.')
        
        logger.info('Num vars for final model: ' + 
                    str(self._numVarsForFinalModel))
        
        logger.info('Min var usage: ' + str(self._minTimesEachVarUsed))
        logger.info('CPUs in use: ' + str(self._numCpus))
        logger.info('Max trials: ' + str(self._maxTrials))
        logger.info('Read ' + str(len(self._trials)) + ' existing trials.')

    # ------------------------------------------------------------------------
    # initXY
    # ------------------------------------------------------------------------
    def _initXY(self, 
                masterTraining: MasterTraining, 
                procTFilesIndependently: bool) -> None:
        
        if not self._xPath.exists() or \
            not self._xTestPath.exists() or \
            not self._xTrainPath.exists() or \
            not self._yPath.exists() or \
            not self._yTestPath.exists() or \
            not self._yTrainPath.exists():
            
            self._logger.info('Writing training to Parquet.')
            
            sampName = masterTraining.dataset.schema.names \
                       [MasterTraining.SAMPLE_COL]

            X, xTrain, xTest, y, yTrain, yTest = \
                self._trainTestSplit(masterTraining, 
                                     sampName, 
                                     procTFilesIndependently)
            
            X.to_parquet(self._xPath)
            xTrain.to_parquet(self._xTrainPath)
            xTest.to_parquet(self._xTestPath)
            
            y.to_frame().to_parquet(self._yPath)
            yTrain.to_frame().to_parquet(self._yTrainPath)
            yTest.to_frame().to_parquet(self._yTestPath)
            
        else:
            self._logger.info('Reading x/y from Parquet as needed.')
            
    # ------------------------------------------------------------------------
    # trainTestSplit
    # ------------------------------------------------------------------------
    def _trainTestSplit(self,
                        masterTraining: MasterTraining,
                        sampName: str, 
                        procTFilesIndependently: bool) -> [pd.DataFrame]:
        
        self._logger.info('Running train-test split.')
        
        X: pd.DataFrame = None
        y: pd.DataFrame = None
        xTrain: pd.DataFrame = None
        yTrain: pd.DataFrame = None
        xTest: pd.DataFrame = None
        yTest: pd.DataFrame = None
        
        if procTFilesIndependently:

            loopX = []
            loopY = []
            loopXTrain = []
            loopYTrain = []
            loopXTest = []
            loopYTest = []
            
            for fName in masterTraining.dataset.files:
                
                xTemp = ParquetDataset(fName).read().to_pandas()
                xTemp = xTemp.replace(-10001, 10001)
                loopX.append(xTemp.drop(sampName, axis=1))
                
                yTemp = xTemp[sampName]
                loopY.append(yTemp)

                xTrainTemp, xTestTemp, yTrainTemp, yTestTemp = \
                    train_test_split(xTemp, yTemp)
                    
                loopXTrain.append(xTrainTemp)
                loopXTest.append(xTestTemp)
                loopYTrain.append(yTrainTemp)
                loopYTest.append(yTestTemp)
                
                del xTemp
                del yTemp
                del xTrainTemp
                del xTestTemp
                del yTrainTemp
                del yTestTemp

            X = pd.concat(loopX)
            y = pd.concat(loopY)
            xTrain = pd.concat(loopXTrain)
            xTest = pd.concat(loopXTest)
            yTrain = pd.concat(loopYTrain)
            yTest = pd.concat(loopYTest)

            del loopX
            del loopY
            del loopXTrain
            del loopYTrain
            del loopXTest
            del loopYTest

        else:

            X = self._masterTraining.dataset.read().to_pandas()
            X = X.replace(-10001, 10001)
            y = X[sampName]
            X.drop(sampName, axis=1, inplace=True)
            xTrain, xTest, yTrain, yTest = train_test_split(X, y)
            
        gc.collect()

        return X, xTrain, xTest, y, yTrain, yTest

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
        return len(self._trials)
        
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
    def run(self) -> None:

        topN = []
        numCompleted = self._numTrials
        varUsageCount = dict.fromkeys(self._allVars, 0)
        rf: RandomForestRegressor = None
        allConditionsMet = False
        averages = {}
        
        while not allConditionsMet:
            
            self._logger.info(str(self.numTrials) + ' completed.')
            
            # This is mostly for testing, so tests complete quickly.
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
            
            self.runFinalModel(topN)
            self._printSortedPredictors(averages)

    # ------------------------------------------------------------------------
    # runTrials
    # ------------------------------------------------------------------------
    def _runTrials(self, numCompleted: int) -> (list, int):
        
        completedTrials = []
        numCompleted = 0
        
        if self._numCpus == 1:
            
            self._logger.info('Starting a batch of 1 trial.')
            trial: Trial = self._runOneTrial()
            completedTrials.append(trial)
            
        else:
            
            with ProcessPoolExecutor(max_workers=self._numCpus) as xtor:

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
        xTrain = pd.read_parquet(self._xTrainPath, columns=colNames)
        yTrain = pd.read_parquet(self._yTrainPath).to_numpy().ravel()
        rf: RandomForestRegressor = self._runRandomForest(xTrain, yTrain)

        X = pd.read_parquet(self._xPath, columns=colNames)
        y = pd.read_parquet(self._yPath).to_numpy().ravel()

        permImportances = permutation_importance(rf, X, y)
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

        rf = RandomForestRegressor(n_estimators=30)
        rf = rf.fit(xTrain, yTrain)
        return rf

    # ------------------------------------------------------------------------
    # runFinalModel
    #
    # Having this as a final model means we can independently choose and
    # specify the top-n predictors.  Specifically, we can use this to create
    # a model based on the top predictors identified in the pge61 code.
    # ------------------------------------------------------------------------
    def runFinalModel(self, colsInFinalModel: list) -> None:
        
        self._logger.info('Running final model.')
        xTrain = pd.read_parquet(self._xTrainPath, columns=colsInFinalModel)
        yTrain = pd.read_parquet(self._yTrainPath).to_numpy().ravel()
        self._rf = self._runRandomForest(xTrain, yTrain)
        self._topN = colsInFinalModel
        
    # ------------------------------------------------------------------------
    # saveFinalModel
    # ------------------------------------------------------------------------
    def saveFinalModel(self) -> None:
        
        outPath: Path = self._outDir / (self._trainingType.value + '.bin')
        topPath: Path = self._outDir / (self._trainingType.value + 'TopN.bin')
        
        with open(outPath, 'wb') as f:
            joblib.dump(self._rf, f)

        with open(topPath, 'wb') as f:
            pickle.dump(self._topN, f)
                