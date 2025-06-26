
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# from modis_vcf.model.Band import Band
from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.ProductType import ProductType
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
from modis_vcf.model.TrainingType import TrainingType


# ----------------------------------------------------------------------------
# BuildTraining
#
# Include all tiles for the given year.
# Training data: /explore/nobackup/projects/ilab/data/MODIS/MODIS_VCF/Mark_training/VCF_training_adjusted/tile_adjustment/v5.0.3samp/
#
# Percent Tree Tiles:  h08v04 h08v05 h09v04 h09v05 h10v04 h10v05 h10v06 h11v02 h11v03 h11v04 h11v05 h11v08 h11v09 h11v10 h12v01 h12v02 h12v03 h12v04 h12v05 h12v09 h12v10 h12v12 h13v01 h13v10 h13v11 h13v12 h16v01 h17v05 h18v03 h18v04 h18v07 h19v04 h19v08 h19v09 h19v10 h19v11 h19v12 h20v02 h20v03 h20v06 h20v08 h20v09 h20v10 h20v11 h21v01 h21v02 h21v04 h21v05 h21v06 h21v10 h22v03 h23v02 h23v03 h24v03 h24v04 h26v06 h27v04 h27v06 h27v07 h29v11 h29v12 h30v12 h31v11
# 
# Percent Bare Tiles:  h08v05 h08v06 h09v05 h09v06 h10v02 h11v02 h11v08 h11v10 h12v01 h12v02 h13v01 h13v02 h16v01 h17v07 h18v07 h18v08 h19v11 h19v12 h20v06 h20v07 h20v09 h21v01 h21v06 h21v07 h21v08 h21v09 h26v03 h26v04 h27v03 h27v04 h29v11 h29v12 h30v11 h30v12 h31v09
#
# TODO: Validate input.
#
# TODO: A Parquet file for a TID could contain a subset of the available 
#       metrics.  When a different set of metrics is requested and this class
#       finds and existing file.  It will stop and return the file, which will
#       not contain what the client requested.  Fix this.
# ----------------------------------------------------------------------------
class BuildTraining(object):
    
    TRAINING_DIR = Path(__file__).parent.parent / 'data' / 'training'
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 year: int, 
                 modisDir: Path,
                 metricsDir: Path,
                 outDir: Path,
                 trainingDir: Path,
                 trainingType: TrainingType=None,
                 tileIds: list = None, 
                 metricNames: list = None,
                 productType: ProductType = None,
                 logger: logging.RootLogger = None):
    
        if not year:
            raise RuntimeException('A year must be provided.')
            
        if not outDir.exists():
            
            raise RuntimeError('Output directory, ' + 
                               str(outDir) + 
                               ', does not exist.')
        
        if not logger:
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

        self._logger: logging.RootLogger = logger

        if (not self._logger.hasHandlers()):

            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            self._logger.addHandler(ch)

        self._year: int = year
        self._modisDir: Path = modisDir
        self._metricsDir: Path = metricsDir
        self._outDir: Path = outDir
        self._metricNames: list = metricNames
        self._productType = productType or ProductTypeMod44(self._modisDir)
        self._outFileSuffix = '-training+obsForRF.parq'
        
        self._trainingType: TrainingType = \
            trainingType or TrainingType.PCT_TREE
        
        self._trainingFilePrefix = \
            'VCF_training_' + self._trainingType.value + '_'
        
        self._trainingFileSuffix = '.7.0.0.out'
        
        self._tids: list = tileIds or self._getTileIds()

        self._logger.info('Year: ' + str(self._year))
        self._logger.info('MODIS dir: ' + str(self._modisDir))
        self._logger.info('Metrics dir: ' + str(self._metricsDir))
        self._logger.info('Output dir: ' + str(self._outDir))
        self._logger.info('Training type: ' + self._trainingType.value)
        self._logger.info('TIDs: ' + str(self._tids))
        self._logger.info('Metric names: ' + str(self._metricNames))
        
    # ------------------------------------------------------------------------
    # addOneMetricToDf
    # ------------------------------------------------------------------------
    def _addOneMetricToDf(self, df: pd.DataFrame, tid: str) -> pd.DataFrame:
        
        mets = Metrics(tid,
                       self._year,
                       self._productType,
                       self._metricsDir,
                       self._logger)

        metricsToRun: list = self._metricNames or mets.availableMetrics

        for metricName in metricsToRun:
            
            # metric: Band = mets.getMetric(metricName)
            metric: list[Metrics.Metric] = mets.getMetric(metricName)
            
            for bandName in metric.dayXref:
                
                index = metric.dayXref[bandName]
                df[bandName] = metric.cube[index].flatten().astype(np.int16)
            
        return df

    # ------------------------------------------------------------------------
    # addTrainingToDf
    # ------------------------------------------------------------------------
    def _addTrainingToDf(self, 
                         df: pd.DataFrame, 
                         tid: str = None) -> pd.DataFrame:
        
        allTraining = []
        tids = [tid] or self._tids
        
        for tid in tids:
            
            tFileName = self.getTrainingFileName(tid)
            samples: np.ndarray = np.fromfile(tFileName, np.uint8)
            
            # Using Band.NO_DATA converts samples from uint8 to int16.
            samples = np.where(samples == 255, Band.NO_DATA, samples)
            allTraining = np.append(allTraining, samples).astype(np.int16)

        # Add the training to the data frame as one big column.
        df[self._trainingType.value] = allTraining
        
        return df
        
    # ------------------------------------------------------------------------
    # getTileIds
    # ------------------------------------------------------------------------
    def _getTileIds(self) -> list:

        sampleFiles = \
            BuildTraining.TRAINING_DIR.glob('*' + self._trainingFileSuffix)
        
        tids = [f.name.split('.')[0] for f in sampleFiles]
        tids = [tid.split('_')[-1] for tid in tids]
        return tids
        
    # ------------------------------------------------------------------------
    # getTrainingFileName
    # ------------------------------------------------------------------------
    def getTrainingFileName(self, tid:str) -> Path:
        
        baseName = self._trainingFilePrefix + tid + self._trainingFileSuffix
        tFileName = BuildTraining.TRAINING_DIR / baseName
            
        return tFileName
        
    # ------------------------------------------------------------------------
    # initializeDataFrame
    # ------------------------------------------------------------------------
    def _initializeDataFrame(self, tid = None) -> pd.DataFrame:
        
        self._logger.info('Initializing data frame.')

        dfRows = {}  # {'tid-x-y': [tid, x, y]}

        tids = [tid] or self._tids
        
        for tid in tids:
        
            self._logger.info('Adding tid ' + tid)

            for x in range(ProductType.COLS):
        
                for y in range(ProductType.ROWS):
        
                    tidYear = tid + '-' + str(self._year)
                    key = tidYear + '-' + str(x) + '-' + str(y)
                    dfRows[key] = [tidYear, x, y]
                    
        self._logger.info('Instantiating data frame.')

        df = pd.DataFrame.from_dict( \
            dfRows,
            orient='index',
            columns=['tid-year', 'x', 'y'])

        return df
        
    # ------------------------------------------------------------------------
    # run
    #
    # Unsorted monthly bands --> 96 metrics
    # 263 metrics
    # Training file name:  Master-yyyy.parquet
    # #cols = #metrics = 
    # col: tileid, x, y, label (from training), m1, ..., mn
    # tile 1 is a set of rows
    # tile 2 appends to that
    #
    # Mark 11/30
    # Each row represents a point
    # Each day is a column
    # Need label column from training data.  Call %tree or whatever the var is.
    # 12 columns for each band x 8 bands = 96 metrics
    # Fixed set of rows and cols, like 10M rows = num pixels in training.
    #
    # DataFrame -> pyarrow.Table -> Parquet
    # https://arrow.apache.org/docs/python/parquet.html
    #
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html
    # Adding a column to a DataFrame is relatively fast. However, adding a row 
    # requires a copy, and may be expensive. We recommend passing a pre-built 
    # list of records to the DataFrame constructor instead of building a 
    # DataFrame by iteratively appending records to it.
    #
    # Training file name:  Master-yyyy.parquet
    # ------------------------------------------------------------------------
    def run(self) -> list:
        
        self._logger.info('Running tiles: ' + str(self._tids))
        
        # Create a file for each tid.
        tidFiles = []
        numComplete = 0
        failedTids = []

        for tid in self._tids:
            
            self._logger.info('Running tile: ' + str(tid))
        
            parqDir = self._outDir
            parqDir.mkdir(exist_ok=True)
            
            outFile = parqDir / \
                      (tid + '-' + 
                       str(self._year) + '-' +
                       self._trainingType +
                       self._outFileSuffix)
                      
            tidFiles.append(outFile)

            if outFile.exists() and self._logger:

                numComplete += 1
                self._logger.info('Parquet file exists: ' + str(outFile))
                continue
                
            # Ensure the training file exists.
            tFileName = self.getTrainingFileName(tid)
            
            if not tFileName.exists():
                
                numComplete += 1

                self._logger.warning('Samples file for ' + 
                                     tid + 
                                     ', ' + 
                                     str(tFileName) +
                                     ' does not exist.')

                failedTids.append(tid)
                continue
            
            # tid-x-y, tid, x, y
            df: pd.DataFrame = self._initializeDataFrame(tid)

            # tid-x-y, tid, x, y, training
            df: pd.DataFrame = self._addTrainingToDf(df, tid)

            # tid-x-y, tid, x, y, training, metric 1, metric 2, ...
            try:
                import pdb
                pdb.set_trace()
                df: pd.DataFrame = self._addOneMetricToDf(df, tid)

            except AttributeError:
                
                failedTids.append(tid)
                self._logger.error('Failed tid ' + str(tid))
                
            # Remove rows that do not have training data.
            df: pd.DataFrame = df[df[self._trainingType.value] != Band.NO_DATA]

            # Data frame to Parquet.
            self._logger.info('Writing ' + str(outFile))
            df.to_parquet(outFile, compression='gzip', index=False)
            numComplete += 1
            
            self._logger.info('Completed ' + 
                              str(numComplete) +
                              ' of ' + 
                              str(len(self._tids)))
                              
        self._logger.warning('Failed tids: ' + str(failedTids))
        
        return tidFiles
