


# ----------------------------------------------------------------------------
# Trial
# ----------------------------------------------------------------------------
class Trial(object):
    
    # ------------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------------
    def __init__(self, 
                 name: str, 
                 predictorNames: list[str],
                 permImportances: dict):
        
        self._name: str = name
        self._permImportances: dict = permImportances
        self._predictorNames: list[str] = predictorNames
        
    # ------------------------------------------------------------------------
    # includesPredictor
    # ------------------------------------------------------------------------
    def includesPredictor(self, predictorName: str) -> bool:
        return predictorName in self.predictorNames
        
    # ------------------------------------------------------------------------
    # index
    # ------------------------------------------------------------------------
    def index(self, predictorName: str) -> int:

        if self.includesPredictor(predictorName):
            return self.predictorNames.index(predictorName)
            
        return -1
        
    # ------------------------------------------------------------------------
    # name
    # ------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name
        
    # ------------------------------------------------------------------------
    # importanceMean
    # ------------------------------------------------------------------------
    def importanceMean(self, predictorName: str) -> float:

        if self.includesPredictor(predictorName):
            
            pos: int = self.index(predictorName)
            return self.permImportances['importances_mean'][pos]
            
        else:
            raise RuntimeError('Trial does not include predictor ' + \
                               predictorName)
        
    # ------------------------------------------------------------------------
    # permImportances
    # ------------------------------------------------------------------------
    @property
    def permImportances(self) -> dict:
        return self._permImportances
        
    # ------------------------------------------------------------------------
    # predictorNames
    # ------------------------------------------------------------------------
    @property
    def predictorNames(self) -> list[str]:
        return self._predictorNames
        
        