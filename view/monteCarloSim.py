#!/usr/bin/python

import argparse
import logging
from pathlib import Path
import sys

from modis_vcf.model.MonteCarloSim import MonteCarloSim


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/monteCarloSim.py --trainingDir /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/MOD44/training --minVarUsage 1 
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
    # Logging
    # ---
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # ---
    # Monte Carlo Simulation
    # ---
    mcs = MonteCarloSim(args.trainingDir)
    mcs.run()


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
