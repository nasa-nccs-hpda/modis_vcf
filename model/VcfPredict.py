
import joblib
import logging
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
from cuml.ensemble import RandomForestRegressor

from osgeo import gdal

from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
from modis_vcf.model.ProductTypeMod44W import ProductTypeMod44W
from modis_water.model.Utils import Utils
 
MOD44_DIR = Path('/css/modis/Collection6.1/L3/MOD44B-VCF/dev')
MOD44W_DIR = Path('/css/modis/Collection6.1/L3/MOD44W-LandWaterMask')


# ----------------------------------------------------------------------------
# VcfPredict
# ----------------------------------------------------------------------------
class VcfPredict(object):
    
    MASK_WATER = 1
    MASK_OUT_OF_PROJ = 250
    PRED_WATER = 200
    PRED_OUT_OF_PROJ = 250

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 treeCoverModelFile: Path,
                 treeCoverTopNFile: Path,
                 nonvegModelFile: Path,
                 nonvegModelTopNFile: Path,
                 outDir: Path,
                 metricsDir: Path,
                 mod44Dir: Path = MOD44_DIR,
                 mod44wDir: Path = MOD44W_DIR,
                 logger: logging.RootLogger = None):
                 
        # Validate the output directory.
        if outDir is None or not outDir.exists() or not outDir.is_dir():
            raise RuntimeError('A valid output directory must be provided.')
            
        self._outDir: Path = outDir
        
        # Load the models.
        self._treeCoverRf: RandomForestRegressor = None
        self._treeCoverTopN: list = None
        
        with open(treeCoverModelFile, 'rb') as f:
            self._treeCoverRf: RandomForestRegressor = joblib.load(f)
            
        with open(treeCoverTopNFile, 'rb') as f:
            self._treeCoverTopN = pickle.load(f)
            
        self._nonvegRf: RandomForestRegressor = None
        self._nonvegTopN: list = None
        
        with open(nonvegModelFile, 'rb') as f:
            self._nonvegRf: RandomForestRegressor = joblib.load(f)
            
        with open(nonvegModelTopNFile, 'rb') as f:
            self._nonvegTopN = pickle.load(f)
            
        # Instantiate the product type for the metrics.
        self._productType = ProductTypeMod44(mod44Dir)
        self._metricsDir: Path = metricsDir
        
        # Instantiate a objects for the water masks.
        self._productTypeM44W = ProductTypeMod44W(mod44wDir)
        self._waterMasks = {}  # {tidyear: water mask}
        
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
    # getMask
    # ------------------------------------------------------------------------
    def _getMask(self, tid: str, year: int) -> np.ndarray:
        
        key = tid + str(year)
        
        if key not in self._waterMasks:
            
            bdf = BandDayFile(). \
                initFromParams(self._productTypeM44W,
                               self._productTypeM44W.WATER_MASK,
                               tid,
                               year,
                               1,
                               self._outDir,
                               self._logger)
        
            band: np.ndarray = bdf.raster(applyQa=False)
            self._waterMasks[key] = band
        
        return self._waterMasks[key]

    # ------------------------------------------------------------------------
    # getMetrics
    # ------------------------------------------------------------------------
    def _getMetrics(self, tid: str, year: int, rfNames: list) -> pd.DataFrame:

        names = rfNames
        
        self._logger.info('Retrieving metrics: ' + str(names))
        
        mInstance = Metrics(tid, 
                            year, 
                            self._productType, 
                            self._metricsDir, 
                            self._logger)
        
        metrics = pd.DataFrame()
        
        for i in range(len(names)):
            metrics[names[i]] = mInstance.getMetricFromRf(names[i]).ravel()

        return metrics
        
    # ------------------------------------------------------------------------
    # maskPrediction
    # ------------------------------------------------------------------------
    def _maskPrediction(self, 
                        tid: str, 
                        year: int, 
                        prediction: np.ndarray) -> np.ndarray:
        
        mask: np.ndarray = self._getMask(tid, year)
        
        maskedPred = np.where(mask == VcfPredict.MASK_WATER,
                              VcfPredict.PRED_WATER, 
                              prediction)
                              
        maskedPred = np.where(mask == VcfPredict.MASK_OUT_OF_PROJ,
                              VcfPredict.PRED_OUT_OF_PROJ, 
                              maskedPred)
        
        return maskedPred
        
    # ------------------------------------------------------------------------
    # runPrediction
    # ------------------------------------------------------------------------
    def _runPrediction(self, 
                       tid: str, 
                       year: int, 
                       rf: RandomForestRegressor,
                       topN: list) -> np.ndarray:
        
        X: pd.DataFrame = self._getMetrics(tid, year, topN)
        # X = X.replace(-10001, 10001)
        
        self._logger.info('Predicting ...')
        
        prediction: np.ndarray = rf.predict(X). \
                                 to_numpy(). \
                                 astype(np.int16). \
                                 reshape(self._productType.ROWS,
                                         self._productType.COLS)

        # prediction: np.ndarray = rf.predict(X). \
        #                          astype(np.int16). \
        #                          reshape(self._productType.ROWS,
        #                                  self._productType.COLS)

        # Apply the water mask.
        maskedPred: np.ndarray = self._maskPrediction(tid, year, prediction)
        
        return maskedPred
        
    # ------------------------------------------------------------------------
    # runTileForYear
    # ------------------------------------------------------------------------
    def runTileForYear(self, tid: str, year: int) -> list[np.ndarray]:

        self._logger.info('Running ' + tid + ' for ' + str(year))
        
        pctTc: np.ndarray = self._runPrediction(tid, 
                                                year, 
                                                self._treeCoverRf,
                                                self._treeCoverTopN)
                                                
        self._write(tid, year, 'Percent_Tree_Cover', pctTc)     

        pctNonveg: np.ndarray = self._runPrediction(tid, 
                                                    year, 
                                                    self._nonvegRf,
                                                    self._nonvegTopN)
                                                    
        self._write(tid, year, 'Percent_NonVegetated', pctNonveg)     

        # ---
        # If 100 - pctTc - pctNonVeg < 0, that the percentages add up to 
        # over 100.  Clamp pctNonTreeVeg to 0 in that case.
        # ---
        zeros = np.zeros_like(pctTc)
        pctNontreeVeg: np.ndarray = np.maximum(0, 100 - pctTc - pctNonveg)
        self._write(tid, year, 'Percent_NonTree_Vegetation', pctNontreeVeg)
        
        return pctTc, pctNonveg, pctNontreeVeg
        
    # ------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------
    def run(self, tids: list[str], years: list[int]) -> \
        list[list[np.ndarray]]:
        
        results = [self.runTileForYear(t, y) for t in tids for y in years]
        return results

    # ------------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------------
    def _write(self, 
               tid: str, 
               year: int, 
               outPrefix: str, 
               raster: np.ndarray) -> Path:
        
        fName: Path = outPrefix + '-' + tid + '-' + str(year)
        
        modSinu = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 ' + \
                  '+datum=WGS84 +units=m +no_defs'
        
        Utils.writeRaster(self._outDir, raster, fName, projection=modSinu)
        