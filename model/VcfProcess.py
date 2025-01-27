
import logging
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from osgeo import gdal

from modis_vcf.model.Band import Band
from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
from modis_water.model.Utils import Utils
 
MOD44_DIR = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')


# ----------------------------------------------------------------------------
# VcfProcess
# ----------------------------------------------------------------------------
class VcfProcess(object):
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
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
    def _getMetrics(self, 
                    tid: str, 
                    year: int, 
                    metName: str = None) -> dict:
        
        names = [metName] if metName else self._rf.feature_names_in_
        self._logger.info('Retrieving metrics: ' + str(names))
        
        mInstance = Metrics(tid, 
                            year, 
                            self._productType, 
                            self._outDir, 
                            self._logger)
        
        # ---
        # I added Juijitsu to Metrics to accommodate this method.  Move that
        # mess here to make Metrics closer to pure.
        # ---
        # metrics = {n: metrics.getMetricFromRf(n).ravel() for n in names}

        metrics = {}
        
        for name in names:
            
            # ---
            # Adjust the metric name to suit metInstance's year, and account
            # for days that wrap to the following year.
            # ---
            parts = name.split('-')
            newName = parts[0] + '-' + parts[1]

            if len(parts) == 3:
                
                modelDay = int(parts[2][-3:])
                adjustedYear = mInstance.getYearForDay(modelDay)
            
                day = 'Day_' + str(adjustedYear) + parts[2][-3:] \
                      if len(parts) == 3 else None
                      
                newName += '-' + day
                
            metrics[newName] = mInstance.getMetricFromRf(newName).ravel()
            
        return metrics
            
    # ------------------------------------------------------------------------
    # runTileForYear
    # ------------------------------------------------------------------------
    def runTileForYear(self, tid: str, year: int) -> Path:
                 
        self._logger.info('Running ' + tid + ' for ' + str(year))
        
        # Get the metrics, as images, for the top n predictors.
        metrics: dict = self._getMetrics(tid, year)

        # Prepare X.
        X: pd.DataFrame = pd.DataFrame.from_dict(metrics)
        X = X.where(X != -10001, 10001)
        
        # Run Random Forest.
        prediction = self._rf.predict(X).reshape(Band.ROWS, Band.COLS)

        # Write the prediction as an image.
        fName: Path = tid + '-' + str(year) + '-predictions.tif'
        
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
        