


# ----------------------------------------------------------------------------
# Trial
# ----------------------------------------------------------------------------
class Trial(object):
    
    # ------------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------------
    def __init__(self, name: str, sampleLocs: list):
        
        self._name: str = name
        
        # ---
        # This is a list of (row, col) coordinates into the matrix of metrics.
        # This represents the randomly-chosen points to feed this trial's 
        # model.
        # ---
        self._sampleCols = []
        
    # ------------------------------------------------------------------------
    # sampleLocs
    # ------------------------------------------------------------------------
    @property
    def sampleLocs(self) -> list:
        return self._sampleLocs
        
        