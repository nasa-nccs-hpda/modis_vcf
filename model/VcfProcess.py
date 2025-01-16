
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
    def _getMetrics(self, 
                    tid: str, 
                    year: int, 
                    metName: str = None) -> dict:
        
        names = [metName] if metName else self._rf.feature_names_in_
        self._logger.info('Retrieving metrics: ' + str(names))
        
        metrics = Metrics(tid, 
                          year, 
                          self._productType, 
                          self._outDir, 
                          self._logger)
        
        metrics = {n: metrics.getMetricFromRf(n).ravel() for n in names}
            
        return metrics
            
    # ------------------------------------------------------------------------
    # runOneTile
    # ------------------------------------------------------------------------
    def runOneTile(self, tid: str, year: int) -> Path:
                 
        # Get the metrics, as images, for the top n predictors.
        metrics: dict = self._getMetrics(tid, year)

        # Prepare X.
        X: pd.DataFrame = pd.DataFrame.from_dict(metrics)
        X = X.where(X != -10001, 10001)
        
        # Run Random Forest.
        prediction = self._rf.predict(X).reshape(Band.ROWS, Band.COLS)

        # Write the prediction as an image.
        outFile: Path = self.writePrediction(prediction, tid, year)

    # ------------------------------------------------------------------------
    # writePrediction
    # ------------------------------------------------------------------------
    def writePrediction(self, 
                        prediction: np.ndarray, 
                        tid: str, 
                        year: int) -> Path:
                        
        fName: Path = self._outDir / \
                      (tid + '-' + str(year) + '-predictions.tif')
                      
        ds = gdal.GetDriverByName('GTiff').Create(
            str(fName),
            prediction.shape[0],
            prediction.shape[1],
            1,
            gdal.GDT_Int16,
            options=['BIGTIFF=YES'])

        ds.SetSpatialRef(Band.modisSinusoidal)
        
        ds.WriteRaster(0, 
                       0, 
                       prediction.shape[0], 
                       prediction.shape[1], 
                       prediction)
                       
        self._logger.info('Wrote ' + str(fName))
