#!/usr/bin/python

import argparse
from pathlib import Path
import sys

from modis_vcf.model.MonteCarloSim import MonteCarloSim


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/monteCarloSim.py --trainingDir /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/MOD44/training --minVarUsage 0 --numTrials 10   # noqa: E501
#
# Timing of the above command.  Training consists of h09v05 metrics.
# ilab207: 4m26.821s, 4m28.300s
# ilab213: 3m15.151s, 3m13.851s
#
# time modis_vcf/view/monteCarloSim.py --trainingDir /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles/MOD44C/training-V5.0.3/  # noqa: E501
# ilab213: 
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to run a Monte Carlo simulation on metrics.'
    parser = argparse.ArgumentParser(description=desc)

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

    parser.add_argument('--trainingDir',
                        required=True,
                        type=Path,
                        help='Directory containing training ' +
                             'data in Parquet files')

    args = parser.parse_args()
    
    # ---
    # Monte Carlo Simulation
    # ---
    mcs = MonteCarloSim(args.trainingDir, 
                        args.numTrials, 
                        args.numPredsPerTrial,
                        args.numVarsForFinalModel,
                        args.minVarUsage)

    topN: list = mcs.run()
    print(topN)
    

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
