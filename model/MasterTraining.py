
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetDataset
from pyarrow.parquet import ParquetSchema


# ----------------------------------------------------------------------------
# MasterTraining
#
# This primary objective of this class is to ensure the schemas of the
# Parquet files match.  It also correctly counts the number of columns, 
# ignoring the ones related to indexing.
# https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html#pyarrow.parquet.ParquetDataset
# https://arrow.apache.org/docs/python/generated/pyarrow.dataset.ParquetFileFragment.html#pyarrow-dataset-parquetfilefragment
# https://arrow.apache.org/docs/python/generated/pyarrow.parquet.FileMetaData.html
# https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetSchema.html#pyarrow.parquet.ParquetSchema
# ----------------------------------------------------------------------------
class MasterTraining(object):
    
    # ---
    # The fist four columns, tid-year, x, y, sample value should be excluded
    # from the random search.
    # ---
    SAMPLE_COL = 3
    START_COL = 4
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, trainingDir: Path, logger: logging.RootLogger):
    
        self._logger: logging.RootLogger = logger
        tFiles = list(trainingDir.glob('*.parq'))
        self._trainingDs: ParquetDataset = ParquetDataset(tFiles)

        # ---
        # As we count the rows below, ensure each fragment's schema is
        # the same.  ParquetDataset aggregates individual Parquet files as a 
        # union, so every column encountered will be in the dataset.  We need 
        # every fragment to have the same columns (which are metrics).
        # Compare each schema to that of the first fragment.
        #
        # Testing firstSchema == None when it is not None causes seg fault!
        # Therefore, just assign it here.
        # ---
        firstSchema: ParquetSchema = \
            self._trainingDs.fragments[0].metadata.schema
        
        for frag in self._trainingDs.fragments:
            
            if not firstSchema.equals(frag.metadata.schema):
                
                msg = 'Parquet ' + str(frag.path) + ' differs from the ' + \
                      'primary schema.'
                      
                raise RuntimeError(msg)
                
        # ---
        # These are the column names from which to randomly select.  Use the
        # names because ParquetDataset.read() returns columns based on names.
        # ---
        self._colNames = firstSchema.names[MasterTraining.START_COL:]

    # ------------------------------------------------------------------------
    # dataset
    # ------------------------------------------------------------------------
    @property
    def dataset(self) -> ParquetDataset:
        return self._trainingDs

    # ------------------------------------------------------------------------
    # toPandas
    #
    # This convenience method could lead to memory-crashing sized data
    # structures.  
    # ------------------------------------------------------------------------
    def toPandas(self) -> pd.DataFrame:
        
        table: pa.Table = self.dataset.read()
        return table.to_pandas()
        
    # ------------------------------------------------------------------------
    # writeCsv
    #
    # This convenience method could lead to memory-crashing sized data
    # structures.  
    # ------------------------------------------------------------------------
    def writeCsv(self, outDir: Path) -> Path:
        
        csvPath = outDir / (Path(self.dataset.files[0]).name + '.csv')
        self.toPandas().to_csv(csvPath)
        return csvPath
        