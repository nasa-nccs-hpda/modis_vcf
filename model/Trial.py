


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
                 permImportance: dict):
        
        self._name: str = name
        self._permImportance: dict = permImportance
        self._predictorNames: list[str] = predictorNames
        
    # ------------------------------------------------------------------------
    # name
    # ------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name
        
    # ------------------------------------------------------------------------
    # permImportance
    # ------------------------------------------------------------------------
    @property
    def permImportance(self) -> dict:
        return self._permImportance
        
    # ------------------------------------------------------------------------
    # predictorNames
    # ------------------------------------------------------------------------
    @property
    def predictorNames(self) -> list[str]:
        return self._predictorNames
        
        