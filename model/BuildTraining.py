
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from modis_vcf.model.Band import Band
from modis_vcf.model.Metrics import Metrics


# ----------------------------------------------------------------------------
# BuildTraining
#
# Include all tiles for the given year.
# Training data: /explore/nobackup/projects/ilab/data/MODIS/MODIS_VCF/Mark_training/VCF_training_adjusted/tile_adjustment/v5.0.3samp/
# ----------------------------------------------------------------------------
class BuildTraining(object):
    
    TRAINING_DIR = Path('/explore/nobackup/projects/ilab/data/' +
                        'MODIS/MODIS_VCF/Mark_training/' +
                        'VCF_training_adjusted/tile_adjustment/v5.0.3samp/')
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 year: int, 
                 modisDir: Path,
                 outDir: Path,
                 logger: logging.RootLogger,
                 trainingName: str, 
                 tileIds: list = None, 
                 metricNames: list = None):
    
        if not year:
            raise RuntimeException('A year must be provided.')
            
        if not outDir.exists():
            
            raise RuntimeError('Output directory, ' + 
                               str(outDir) + 
                               ', does not exist.')
        
        self._outDir: Path = outDir
        self._year: int = year
        self._modisDir: Path = modisDir
        self._logger: logging.RootLogger = logger
        self._trainingName: str = trainingName
        self._tids: list = tileIds or BuildTraining._getTileIds()
        self._metricNames: list = metricNames
        self._training = None
        
    # ------------------------------------------------------------------------
    # addMetricsToDf
    # ------------------------------------------------------------------------
    def _addMetricsToDf(self, df: pd.DataFrame) -> pd.DataFrame:
        
        tidMets: dict = self._getAllMetrics()
        
        metricsToRun: list = self._metricNames or \
                             list(tidMets.values())[0].availableMetrics

        for metName in metricsToRun:

            # ---
            # Get the metric for the first tid, so we can know the names
            # of all the bands it includes.
            # ---
            bandNames = list(tidMets.values())[0].getMetric(metName).dayXref
            
            for bandName in bandNames:
                
                self._logger.info('Adding column for ' + bandName)
             
                metCol = []
            
                for tid in self._tids:
                
                    metric: Band = tidMets[tid].getMetric(metName)
                    index = metric.dayXref[bandName]
                    
                    # The default behavior uses float64, so force int16.
                    vals = metric.cube[index].flatten()
                    metCol = np.append(metCol, vals).astype(np.int16)
                    
                df[bandName] = metCol
        
        return df
        
    # ------------------------------------------------------------------------
    # addOneMetricToDf
    # ------------------------------------------------------------------------
    def _addOneMetricToDf(self, df: pd.DataFrame, tid: str) -> pd.DataFrame:
        
        mets = Metrics(tid,
                       self._year,
                       self._modisDir,
                       self._outDir,
                       self._logger)

        metricsToRun: list = self._metricNames or mets.availableMetrics

        for metricName in metricsToRun:
            
            metric: Band = mets.getMetric(metricName)
            
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
            
            # tFileName = BuildTraining.TRAINING_DIR / (tid + '.samp.bin')
            tFileName = self.getTrainingFileName(tid)
            samples: np.ndarray = np.fromfile(tFileName, np.uint8)
            
            # Using Band.NO_DATA converts samples from uint8 to int16.
            samples = np.where(samples == 255, Band.NO_DATA, samples)
            allTraining = np.append(allTraining, samples).astype(np.int16)

        # Add the training to the data frame as one big column.
        df[self._trainingName] = allTraining
        
        return df
        
    # ------------------------------------------------------------------------
    # getParquetName
    # ------------------------------------------------------------------------
    def getParquetName(self) -> Path:
        
        outFile = self._outDir / ('Master-' + str(self._year) + '.parquet')
        return outFile
        
    # ------------------------------------------------------------------------
    # getTileIds
    # ------------------------------------------------------------------------
    @staticmethod
    def _getTileIds() -> list:

        sampleFiles = BuildTraining.TRAINING_DIR.glob('*.samp.bin')
        tids = [f.name.split('.')[0] for f in sampleFiles]
        return tids
        
    # ------------------------------------------------------------------------
    # getTrainingFileName
    # ------------------------------------------------------------------------
    def getTrainingFileName(self, tid) -> Path:
        
        tFileName = BuildTraining.TRAINING_DIR / (tid + '.samp.bin')
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

            for x in range(Band.COLS):
        
                for y in range(Band.ROWS):
        
                    tidYear = tid + '-' + str(self._year)
                    key = tidYear + '-' + str(x) + '-' + str(y)
                    dfRows[key] = [tidYear, x, y]
                    
        self._logger.info('Instantiating data frame.')

        df = pd.DataFrame.from_dict( \
            dfRows,
            orient='index',
            columns=['tid-year', 'x', 'y'],
            dtype=np.int16)

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
    def run(self) -> None:
        
        self._logger.info('Running tiles: ' + str(self._tids))
        
        # Create a file for each tid.
        tidFiles = []
        numComplete = 0
        failedTids = []

        for tid in self._tids:
            
            self._logger.info('Running tile: ' + str(tid))
        
            outFile = self._outDir / (tid + '-' + str(self._year) + '.parq')
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
                                     ' does not exist.')
                continue
            
            # tid-x-y, tid, x, y
            df: pd.DataFrame = self._initializeDataFrame(tid)

            # tid-x-y, tid, x, y, training
            df: pd.DataFrame = self._addTrainingToDf(df, tid)

            # tid-x-y, tid, x, y, training, metric 1, metric 2, ...
            try:
                df: pd.DataFrame = self._addOneMetricToDf(df, tid)

            except AttributeError:
                
                failedTids.append(tid)
                self._logger.error('Failed tid ' + str(tid))
                
            # Remove rows that do not have training data.
            df: pd.DataFrame = df[df[self._trainingName] != Band.NO_DATA]

            # Data frame to Parquet.
            self._logger.info('Writing ' + str(outFile))
            df.to_parquet(outFile, compression='gzip', index=False)
            numComplete += 1
            
            self._logger.info('Completed ' + 
                              str(numComplete) +
                              ' of ' + 
                              str(len(self._tids)))
                              
        self._logger.warn('Failed tids: ' + str(failedTids))

    # ------------------------------------------------------------------------
    # statistics
    #
    # This is a convenience method for development and testing.
    # ------------------------------------------------------------------------
    def statistics(self) -> None:
        
        # Rows that are not full of no-data values.
        print('Total rows:', self.training.shape[0])
        
        numAllNoData = \
            self.training.value_counts( \
                subset=self.training.columns[4:].to_list()) \
                    [Band.NO_DATA].to_list()[-1]
        
        print('Num rows that are full of no-data values:', numAllNoData)
        
    # ------------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------------
    @property
    def training(self):
        
        if self._training is None:
            
            parquetName = self.getParquetName()

            if parquetName.exists():
                self._training = pd.read_parquet(parquetName)
                
            else:
                self.run()
                
        return self._training
    