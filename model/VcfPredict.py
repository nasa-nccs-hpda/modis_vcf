
import joblib
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from osgeo import gdal

from modis_vcf.model.Band import Band
from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
from modis_water.model.Utils import Utils
 
MOD44_DIR = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')


# ----------------------------------------------------------------------------
# VcfPredict
#
# TODO: Should MOD44_DIR be refactored and shared among VCF applications?
# TODO: One more thing.  The answer to "predict" should NEVER be "NoData" if it is then either there is a problem with the metrics or there is a problem with the model.  I say this because I am seeing NoData in the predict results I am looking at from the Notebook and hoping that you aren't seeing any in your results.  Specifically I am looking at tile h16v01 which has the Greenland ice sheets as all NoData.
# ----------------------------------------------------------------------------
class VcfPredict(object):
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 modelFile: Path,
                 outDir: Path,
                 metricsDir: Path,
                 mod44Dir: Path = MOD44_DIR,
                 logger: logging.RootLogger = None):
                 
        # Validate the output directory.
        if outDir is None or not outDir.exists() or not outDir.is_dir():
            raise RuntimeError('A valid output directory must be provided.')
            
        self._outDir: Path = outDir
        
        # Load the model.
        self._rf: RandomForestClassifier = None
        
        with open(modelFile, 'rb') as f:
            self._rf: RandomForestRegressor = joblib.load(f)
            
        # Instantiate the product type for the metrics.
        self._productType = ProductTypeMod44(mod44Dir)
        self._metricsDir: Path = metricsDir
        
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
    def _getMetrics(self, 
                    tid: str, 
                    year: int, 
                    metName: str = None) -> np.ndarray:
        
        names = [metName] if metName else self._rf.feature_names_in_
        self._logger.info('Retrieving metrics: ' + str(names))
        
        mInstance = Metrics(tid, 
                            year, 
                            self._productType, 
                            self._metricsDir, 
                            self._logger)
        
        # ---
        # I added Juijitsu to Metrics to accommodate getMetricFromRf.  Move
        # that mess here to make Metrics closer to pure.
        # ---
        metrics = pd.DataFrame()
        
        for i in range(len(names)):
            metrics[names[i]] = mInstance.getMetricFromRf(names[i]).ravel()

        return metrics
        
    # ------------------------------------------------------------------------
    # runTileForYear
    # ------------------------------------------------------------------------
    def runTileForYear(self, tid: str, year: int) -> Path:
                 
        self._logger.info('Running ' + tid + ' for ' + str(year))
        
        # Get the metrics, as images, for the top n predictors.
        X = self._getMetrics(tid, year)
        X = X.replace(-10001, 10001)
        
        # Run Random Forest.
        prediction = self._rf.predict(X). \
                     astype(np.int16). \
                     reshape(Band.ROWS, Band.COLS)

        # Write the prediction as an image.
        fName: Path = tid + '-' + str(year) + '-predictions'
        
        modSinu = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 ' + \
                  '+datum=WGS84 +units=m +no_defs'
        
        Utils.writeRaster(self._outDir, prediction, fName, projection=modSinu)
                                          
        return fName
        
    # ------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------
    def run(self, tids: list[str], years: list[int]) -> list[Path]:
        
        paths = [self.runTileForYear(t, y) for t in tids for y in years]
        return paths
        