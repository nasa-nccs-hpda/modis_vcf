
import logging

from cuml import RandomForestClassifier

from modis_vcf.model.MonteCarloSim import MonteCarloSim


# ----------------------------------------------------------------------------
# MonteCarloSimGpu
# ----------------------------------------------------------------------------
class MonteCarloSimGpu(MonteCarloSim):
    
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
        

        super(MonteCarloSimGpu, self).__init__(trainingDir,
                                               numTrials,
                                               predictorsPerTrial,
                                               numVarsForFinalModel,
                                               minTimesEachVarUsed,
                                               logger)

    # ------------------------------------------------------------------------
    # runRandomForest
    #
    # TODO: Define return type.
    # ------------------------------------------------------------------------
    def _runRandomForest(self, xTrain, xTest, yTrain, yTest) -> None:
        
        rf = RandomForestClassifier(n_estimators=1)
        rf.fit(X, y)
        return rf
        