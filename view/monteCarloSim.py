#!/usr/bin/python

import argparse
from pathlib import Path
import sys

from modis_vcf.model.MonteCarloSim import MonteCarloSim


# -----------------------------------------------------------------------------
# main
#
# python modis_vcf/view/monteCarloSim.py --trainingDir /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/MOD44/training -o /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/MOD44 --minVarUsage 0 --numTrials 10   # noqa: E501
#
# Timing of the above command.  Training consists of h09v05 metrics.
# ilab207: 4m26.821s, 4m28.300s
# ilab213: 3m15.151s, 3m13.851s
#
# python modis_vcf/view/monteCarloSim.py --trainingDir /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/MOD44/training -o /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/MOD44 --minVarUsage 0 --numTrials 10 --gpu  # noqa: E501
# GPU: 1m2.741s
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to run a Monte Carlo simulation on metrics.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpu',
                        action='store_true',
                        help='Use GPUs')

    parser.add_argument('--minVarUsage',
                        type=int,
                        help='The minimum times each variable ' +
                             'must be used in a trial')

    parser.add_argument('--numPredsPerTrial',
                        type=int,
                        help='The number of predictors per trials')

    parser.add_argument('--numTrials',
                        type=int,
                        help='The number of trials to run')

    parser.add_argument('--numVarsForFinalModel',
                        type=int,
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
    if args.gpu:
        
        # ---
        # Import here to avoid potential errors when MonteCarloSimGpu attempts
        # to import CUML on systems that do not have it installed.
        # ---
        from modis_vcf.model.MonteCarloSimGpu import MonteCarloSimGpu
        
        mcs = MonteCarloSimGpu(args.trainingDir, 
                               args.numTrials, 
                               args.numPredsPerTrial,
                               args.numVarsForFinalModel,
                               args.minVarUsage)

    else:
        
        mcs = MonteCarloSim(args.trainingDir, 
                            args.numTrials, 
                            args.numPredsPerTrial,
                            args.numVarsForFinalModel,
                            args.minVarUsage)

    mcs.saveFinalModel(args.o)


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
