#!/usr/bin/python

import argparse
import multiprocessing
from pathlib import Path
import sys

from sklearn.ensemble import RandomForestRegressor

from modis_vcf.model.MonteCarloSim import MonteCarloSim


# -----------------------------------------------------------------------------
# main
#
# TODO: Remove GPU junk.
# TODO: Should --gpu and --numCpus be mutually exclusive?
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to run a Monte Carlo simulation on metrics.'
    parser = argparse.ArgumentParser(description=desc)

    # parser.add_argument('--gpu',
    #                     action='store_true',
    #                     help='Use GPUs')

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

    # parser.add_argument('--numTrials',
    #                     type=int,
    #                     help='The number of trials to run')

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

    args = parser.parse_args()
    
    # ---
    # Monte Carlo Simulation
    # ---
    # if args.gpu:
    #
    #     # ---
    #     # Import here to avoid potential errors when MonteCarloSimGpu attempts
    #     # to import CUML on systems that do not have it installed.
    #     # ---
    #     from modis_vcf.model.MonteCarloSimGpu import MonteCarloSimGpu
    #
    #     mcs = MonteCarloSimGpu(args.trainingDir,
    #                            args.o,
    #                            args.numTrials,
    #                            args.numPredsPerTrial,
    #                            args.numVarsForFinalModel,
    #                            args.minVarUsage)
    #
    # else:

    numCpus = min(args.numCpus, multiprocessing.cpu_count())
    
    mcs = MonteCarloSim(args.trainingDir, 
                        args.o,
                        # args.numTrials,
                        args.numPredsPerTrial,
                        args.numVarsForFinalModel,
                        args.minVarUsage,
                        numCpus=args.numCpus)

    finalModel: RandomForestRegressor = mcs.run()
    mcs.saveFinalModel(finalModel)


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
