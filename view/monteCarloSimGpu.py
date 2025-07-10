#!/usr/bin/python

import argparse
import multiprocessing
from pathlib import Path
import sys

from sklearn.ensemble import RandomForestRegressor

from modis_vcf.model.MonteCarloSimGpu import MonteCarloSimGpu
from modis_vcf.model.TrainingType import TrainingType


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/monteCarloSimGpu.py --trainingDir /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/SystemTestsGpu/2-Training -o /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/SystemTestsGpu/3-Models --numCpus 1 --trainingType pcttree
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to run a Monte Carlo simulation on metrics.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--minVarUsage',
                        type=int,
                        help='The minimum times each variable ' +
                             'must be used in a trial')

    parser.add_argument('--numCpus',
                        type=int,
                        default=1,
                        help='The number of CPUs to use.  Set to a high \
                              number to use all available CPUs.')

    parser.add_argument('--numPredsPerTrial',
                        type=int,
                        default=10,
                        help='The number of predictors per trials')

    parser.add_argument('--numVarsForFinalModel',
                        type=int,
                        default=20,
                        help='The number of top predictors ' +
                             'to use in the final model')

    parser.add_argument('-o',
                        type=Path,
                        required='True',
                        help='Write the final model here.')

    parser.add_argument('--trainingDir',
                        required=True,
                        type=Path,
                        help='Directory containing training ' +
                             'data in Parquet files')

    parser.add_argument('--trainingType',
                        type=TrainingType,
                        choices=[t.value for t in TrainingType],
                        help='Choose the training type to run.')

    args = parser.parse_args()
    
    mcs = MonteCarloSimGpu(args.trainingDir, 
                           args.trainingType,
                           args.o,
                           args.numPredsPerTrial,
                           args.numVarsForFinalModel,
                           args.minVarUsage)

    mcs.run()
    mcs.saveFinalModel()


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
