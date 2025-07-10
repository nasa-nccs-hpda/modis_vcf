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
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to run a Monte Carlo simulation on metrics.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-o',
                        type=Path,
                        required='True',
                        help='Write the final model here.')

    parser.add_argument('--preds',
                        type=str,
                        nargs='*',
                        help='A space-separated list of predictor names')

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
                           args.o)

    mcs.runFinalModel(args.preds)
    mcs.saveFinalModel()


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
