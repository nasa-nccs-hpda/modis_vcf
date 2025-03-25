
import joblib
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from osgeo import gdal

from modis_vcf.model.Band import Band
from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
from modis_vcf.model.ProductTypeMod44W import ProductTypeMod44W
from modis_water.model.Utils import Utils
 
# MOD44_DIR = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')
MOD44_DIR = Path('/css/modis/Collection6.1/L3/MOD44B-VCF/dev')
MOD44W_DIR = Path('/css/modis/Collection6.1/L3/MOD44W-LandWaterMask')


# ----------------------------------------------------------------------------
# VcfPredict
#
# TODO: Should MOD44_DIR be refactored and shared among VCF applications?
# TODO: One more thing.  The answer to "predict" should NEVER be "NoData" if it is then either there is a problem with the metrics or there is a problem with the model.  I say this because I am seeing NoData in the predict results I am looking at from the Notebook and hoping that you aren't seeing any in your results.  Specifically I am looking at tile h16v01 which has the Greenland ice sheets as all NoData.
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

        # Alter the RF's names to match the year being predicted.
        names = []
        
        for rfName in rfNames:
            
            name, band, day = rfName.split('-')
            prefix, jul = day.split('_')
            newDay = prefix + '-' + str(year) + jul[-3:]
            names.append(newDay)
        
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
                       rf: RandomForestRegressor) -> Path:
        
        X = self._getMetrics(tid, year, rf.feature_names_in_)
        X = X.replace(-10001, 10001)
        
        # Perform the tree cover predictions.
        prediction: np.ndarray = rf.predict(X). \
                                 astype(np.int16). \
                                 reshape(Band.ROWS, Band.COLS)

        # Apply the water mask.
        maskedPred: np.ndarray = self._maskPrediction(tid, year, prediction)
        
        # Write the prediction as an image.
        fName: Path = 'Percent_Tree_Cover' + '-' + tid + '-' + str(year)
        
        modSinu = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 ' + \
                  '+datum=WGS84 +units=m +no_defs'
        
        Utils.writeRaster(self._outDir, maskedPred, fName, projection=modSinu)
                                          
        return fName
        
    # ------------------------------------------------------------------------
    # runTileForYear
    # ------------------------------------------------------------------------
    def runTileForYear(self, tid: str, year: int) -> (Path, Path):

        self._logger.info('Running ' + tid + ' for ' + str(year))
        
        # Get the metrics, as images, for the top n predictors.
        # X = self._getMetrics(tid, year, self._treeCoverRf)
        # X = X.replace(-10001, 10001)
        #
        # # Perform the tree cover predictions.
        # prediction: np.ndarray = self._treeCoverRf.predict(X). \
        #                          astype(np.int16). \
        #                          reshape(Band.ROWS, Band.COLS)
        #
        # # Apply the water mask.
        # maskedPred = self._maskPrediction(tid, year, prediction)
        #
        # # Write the prediction as an image.
        # tCoverPath: Path = 'Percent_Tree_Cover' + '-' + tid + '-' + str(year)
        #
        # modSinu = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 ' + \
        #           '+datum=WGS84 +units=m +no_defs'
        #
        # Utils.writeRaster(self._outDir,
        #                   maskedPred,
        #                   tCoverPath,
        #                   projection=modSinu)
        
        tCoverPath: Path = self._runPrediction(tid, year, self._treeCoverRf)
        nonvegPath: Path = self._runPrediction(tid, year, self._nonvegRf)

        return (tCoverPath, nonvegPath)
        
    # ------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------
    def run(self, tids: list[str], years: list[int]) -> list[Path]:
        
        paths = [self.runTileForYear(t, y) for t in tids for y in years]
        return paths
        