
import logging
from pathlib import Path
import pickle

from sklearn.ensemble import RandomForestClassifier

from modis_vcf.model.Band import Band
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44

MOD44_DIR = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')


# ----------------------------------------------------------------------------
# VcfProcess
# ----------------------------------------------------------------------------
class VcfProcess(object):
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 years: list[int],
                 tids: list[str],
                 modelFile: Path,
                 outDir: Path,
                 mod44Dir: Path = MOD44_DIR,
                 logger: logging.RootLogger = None):
                 
        # Validate the output directory.
        if outDir is None or not outDir.exists() or not outDir.is_dir():
            raise RuntimeError('A valid output directory must be provided.')
            
        self._outDir: Path = outDir
        
        # Load the model.
        self._rf: RandomForestClassifier = None
        
        with open(modelFile, 'rb') as f:
            self._rf: RandomForestClassifier = pickle.load(f)
            
        # Instantiate the product type for the metrics.
        self._productType = ProductTypeMod44(mod44Dir)
        
        # Instantiate the logger.
        if not logger:
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            if (not logger.hasHandlers()):

                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                logger.addHandler(ch)

        self._logger: logging.RootLogger = logger

    # ------------------------------------------------------------------------
    # getMetrics
    # ------------------------------------------------------------------------
    def _getMetrics(self, tid: str, year: int) -> list[Path]:
        
        metrics = Metrics(tid, 
                          year, 
                          self._productType, 
                          self._outDir, 
                          self._logger)
        
        # array(['UnsortedMonthlyBands-Band_1-Day_2019321',
        # 'BandReflMedianGreenness-Band_4', 'Lowest6MeanBandRefl-Band_7',
        # 'BandReflMedian-Band_7', 'Greenest8MeanBandRefl-Band_4',
        # 'Lowest3MeanBandRefl-Band_4',
        # 'UnsortedMonthlyBands-Band_3-Day_2019193',
        # 'UnsortedMonthlyBands-Band_6-Day_2019129',
        # 'UnsortedMonthlyBands-Band_7-Day_2019193',
        # 'UnsortedMonthlyBands-Band_3-Day_2020033',
        # 'Greenest3MeanBandRefl-Band_3', 'Greenest6MeanBandRefl-NDVI',
        # 'UnsortedMonthlyBands-Band_6-Day_2019193', 'BandReflMax-Band_6',
        # 'UnsortedMonthlyBands-Band_1-Day_2019225',
        # 'AmpWarmestBandRefl-Band_7',
        # 'UnsortedMonthlyBands-Band_4-Day_2019289',
        # 'AmpWarmestBandRefl-Band_6',
        # 'UnsortedMonthlyBands-Band_4-Day_2019129',
        # 'BandReflMinGreenness-Band_3'], dtype=object)
        # ndarray
        
        for metName:str in self._rf.feature_names_in_:
            
            nameComponents = metName.split('-')
            baseName = nameComponents[0]
            bandName = nameComponents[1]
            
            day = nameComponents[2].split('_')[1] \
                  if len(nameComponents) > 2 else None
            
            fullMetric: Band =  metrics.getMetric(baseName)
            
            # ---
            # RF needs a 4800 x 4800 raster representation of metrics.  The
            # daily metrics, those with "Day_" in their feature names, must
            # have the day extracted from the full metric, which contains an
            # entire year of metrics.
            # ---
            
        
    # ------------------------------------------------------------------------
    # runOneTile
    # ------------------------------------------------------------------------
    def runOneTile(self, tid: str, year: int) 
                 
        # Get the metrics, as images, for the top n predictors.


    