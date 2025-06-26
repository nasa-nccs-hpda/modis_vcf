
import joblib
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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
                 nonvegModelFile: Path,
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
        
        with open(treeCoverModelFile, 'rb') as f:
            self._treeCoverRf: RandomForestRegressor = joblib.load(f)
            
        self._nonvegRf: RandomForestRegressor = None
        
        with open(nonvegModelFile, 'rb') as f:
            self._nonvegRf: RandomForestRegressor = joblib.load(f)
            
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

        # ---
        # Alter the RF's names to match the year being predicted, if an
        # unsorted monthly band is encountered.
        # ---
        names = []
        
        for rfName in rfNames:
            
            if rfName.count('-') == 2:

                name, band, day = rfName.split('-')
                prefix, jul = day.split('_')
                newDay = prefix + '_' + str(year) + jul[-3:]
                rfName = name + '-' + band + '-' + newDay
                
            names.append(rfName)
        
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
                       rf: RandomForestRegressor) -> np.ndarray:
        
        X: pd.DataFrame = self._getMetrics(tid, year, rf.feature_names_in_)
        X = X.replace(-10001, 10001)
        
        prediction: np.ndarray = rf.predict(X). \
                                 astype(np.int16). \
                                 reshape(self._productType.ROWS,
                                         self._productType.COLS)

        # Apply the water mask.
        maskedPred: np.ndarray = self._maskPrediction(tid, year, prediction)
        
        return maskedPred
        
    # ------------------------------------------------------------------------
    # runTileForYear
    # ------------------------------------------------------------------------
    def runTileForYear(self, tid: str, year: int) -> list(np.ndarray):

        self._logger.info('Running ' + tid + ' for ' + str(year))
        
        pctTc: np.ndarray = self._runPrediction(tid, year, self._treeCoverRf)
        self._write(tid, year, 'Percent_Tree_Cover', pctTc)     

        pctNonVeg: np.ndarray = self._runPrediction(tid, year, self._nonvegRf)
        self._write(tid, year, 'Percent_NonVegetated', pctNonVe)     

        # ---
        # If 100 - pctTc - pctNonVeg < 0, that the percentages add up to 
        # over 100.  Clamp pctNonTreeVeg to 0 in that case.
        # ---
        pctNonTreeVeg: np.ndarray = max(0, 100 - pctTc - pctNonVeg)
        self._write(tid, year, 'Percent_NonTree_Vegetation', pctNonTreeVeg)
        
        return pctTc, pctNonVeg, pctNonTreeVeg
        
    # ------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------
    def run(self, tids: list[str], years: list[int]) -> \
        list[list(np.ndarray)]:
        
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
        