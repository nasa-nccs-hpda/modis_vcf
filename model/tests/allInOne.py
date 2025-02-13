#!/usr/bin/python

from pathlib import Path
import sys

import dask
import cudf
from cuml.dask.ensemble import RandomForestClassifier
from cuml.dask.common import utils as dask_utils
from dask_cuda import LocalCUDACluster
import dask_cudf
import dask.dataframe as dd
from dask.distributed import Client
   

# https://docs.rapids.ai/api/dask-cuda/nightly/troubleshooting/

# Important note: the output from the regressor should be "integer" as the input variables are also integers.  We do not need the output to be "float" because we don't care about the decimals and we don't have enough information in the input data to reliably predict decimals anyway. 

def main():        

    cluster = LocalCUDACluster()
    client = Client(cluster)
    workers = client.has_what().keys()
    numWorkers = len(workers)

    trainingDir = Path('/explore/nobackup/projects/ilab/projects/' \
                       'MODIS-VCF/processedTiles/MOD44C/training-V5.0.3')

    colNames = ['BandReflMax-NDVI',
                'UnsortedMonthlyBands-Band_5-Day-2019321',
                'Lowest3MeanBandRefl-Band_5',
                'BandReflMin-Band_2',
                'BandReflMax-Band_7',
                'UnsortedMonthlyBands-Band_7-Day-2019353',
                'BandReflMaxTemp-Band_2',
                'UnsortedMonthlyBands-Band_3-Day-2019321',
                'BandReflMin-NDVI',
                'AmpBandRefl-Band_6']

    import pdb
    pdb.set_trace()

    X: dd.DataFrame = dd.read_parquet(trainingDir, colNames)
    y: dd.Series = dd.read_parquet(trainingDir, 'PercentTree')

    rf = RandomForestClassifier(n_estimators=numWorkers,
                                client=client,
                                verbose=True)

    # > /usr/local/lib/python3.10/dist-packages/cuml/dask/ensemble/base.py(102)_fit()
    # -> data = DistributedDataHandler.create(dataset, client=self.client)

    # /usr/local/lib/python3.10/dist-packages/distributed/client.py(2185)submit()

    rf.fit(X, y)
    

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
