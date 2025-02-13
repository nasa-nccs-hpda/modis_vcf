
from pathlib import Path
import logging

import pandas as pd
import cudf
# from cuml.model_selection import train_test_split
# from cuml import RandomForestClassifier
# import cupy  # Used?

from modis_vcf.model.MonteCarloSim import DEFAULT_TOP_N
from modis_vcf.model.MonteCarloSim import DEFAULT_VALUE
from modis_vcf.model.MonteCarloSim import MonteCarloSim


# ----------------------------------------------------------------------------
# MonteCarloSimGpu
# ----------------------------------------------------------------------------
class MonteCarloSimGpu(MonteCarloSim):
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 trainingDir: Path, 
                 numTrials: int = DEFAULT_VALUE, 
                 predictorsPerTrial: int = DEFAULT_VALUE, 
                 numVarsForFinalModel: int = DEFAULT_TOP_N,
                 minTimesEachVarUsed: int = DEFAULT_VALUE,
                 logger: logging.RootLogger = None):
        

        super(MonteCarloSimGpu, self).__init__(trainingDir,
                                               numTrials,
                                               predictorsPerTrial,
                                               numVarsForFinalModel,
                                               minTimesEachVarUsed,
                                               logger)

    # ------------------------------------------------------------------------
    # runRandomForest
    #
    # TODO: declare return type
    # ------------------------------------------------------------------------
    def _runRandomForest2(self, 
                          xTrain: pd.core.frame.DataFrame, 
                          yTrain: pd.core.frame.DataFrame):
    
        from cuml.dask.common import utils
        from cuml.dask.ensemble import RandomForestClassifier
        import dask_cudf
        from dask.distributed import Client

        # ValueError: n_estimators cannot be lower than number of dask workers.
        client = Client()
        workers = client.has_what().keys()
        numWorkers = len(workers)
    
        xtCudf = dask_cudf.from_cudf(xTrain, npartitions=numWorkers)
        ytPd = pd.DataFrame(yTrain)
        # ytCudf = dask_cudf.from_cudf(ytPd, npartitions=numWorkers)
        ytCudf = dask_cudf.from_cudf(ytPd[0], npartitions=numWorkers)
    
        # /usr/local/lib/python3.10/dist-packages/cuml/dask/common/utils.py:142: UserWarning: Sending large graph of size 1.01 GiB.
        # This may cause some slowdown.
        # Consider loading the data with Dask directly
        #  or using futures or delayed objects to embed the data into the graph without repetition.
        xt = xtCudf
        yt = ytCudf
        utils.persist_across_workers(client, [xtCudf, ytCudf],  workers=workers)

        rf = RandomForestClassifier(n_estimators=numWorkers, verbose=True)

        # RuntimeError: 5 of 5 worker jobs failed: 'NoneType' object has no attribute 'shape', 'NoneType' object has no attribute
        import pdb
        pdb.set_trace()
        rf.fit(xt, yt)
    


        return rf
    
    # ------------------------------------------------------------------------
    # runRandomForest
    #
    # TODO: declare return type
    # ------------------------------------------------------------------------
    def _runRandomForest(self, 
                         xTrain: pd.core.frame.DataFrame, 
                         yTrain: pd.core.frame.DataFrame):
    
        import dask
        import cudf
        from cuml.dask.ensemble import RandomForestClassifier
        from cuml.dask.common import utils as dask_utils
        from dask_cuda import LocalCUDACluster
        import dask_cudf
        import dask.dataframe as dd
        from dask.distributed import Client
        from dask.distributed import LocalCluster
        from modis_vcf.model.MasterTraining import MasterTraining
        
        import pdb
        pdb.set_trace()
        # cluster = LocalCUDACluster()
        cluster = LocalCluster()
        client = Client(cluster)
        # client = Client()
        workers = client.has_what().keys()
        numWorkers = len(workers)

        # ---
        # Read X and y to Dask, without Pandas.
        # https://docs.dask.org/en/stable/best-practices.html#load-data-with-dask
        # https://docs.dask.org/en/stable/generated/dask.dataframe.read_parquet.html#dask.dataframe.read_parquet
        # ---
        colNames: list[str]  = self._chooseColumns()
        X: dd.DataFrame = dd.read_parquet(self._trainingDir, colNames)

        sampName = self.masterTraining.dataset.schema.names \
                   [MasterTraining.SAMPLE_COL]

        y: dd.Series = dd.read_parquet(self._trainingDir, sampName)

        # pandas.errors.IndexingError: Too many indexers
        # y = y[:10000]
        
        rf = RandomForestClassifier(n_estimators=numWorkers, 
                                    client=client, 
                                    verbose=True)

        # 5 of 5 worker jobs failed: 'NoneType' object has no attribute 'shape'
        rf.fit(X, y)



