#!/usr/bin/python

import argparse
import multiprocessing
from pathlib import Path
import sys

from sklearn.ensemble import RandomForestRegressor

from modis_vcf.model.MonteCarloSim import MonteCarloSim
from modis_vcf.model.TrainingType import TrainingType


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/rfFitFromPge61.py 
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to run a Monte Carlo simulation on metrics.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-o',
                        type=Path,
                        required='True',
                        help='Write the final model here.')

    parser.add_argument('--trainingType',
                        type=TrainingType,
                        choices=[t.value for t in TrainingType],
                        help='Choose the training type to run.')

    args = parser.parse_args()

    treeCoverPreds = [
        'Unsorted MonthlyBands-Band_1-Day_1',
        'Unsorted MonthlyBands-Band_1-Day_2',
        'Unsorted MonthlyBands-Band_1-Day_3',
        'Unsorted MonthlyBands-Band_1-Day_4',
        'Unsorted MonthlyBands-Band_1-Day_5',
        'Unsorted MonthlyBands-Band_1-Day_6',
        'Unsorted MonthlyBands-Band_1-Day_7',
        'Unsorted MonthlyBands-Band_1-Day_9',
        'Unsorted MonthlyBands-Band_2-Day_1',
        'Unsorted MonthlyBands-Band_2-Day_2',
        'Unsorted MonthlyBands-Band_2-Day_4',
        'Unsorted MonthlyBands-Band_2-Day_6',
        'Unsorted MonthlyBands-Band_2-Day_7',
        'Unsorted MonthlyBands-Band_5-Day_7',
        'Unsorted MonthlyBands-Band_5-Day_9',
        'Unsorted MonthlyBands-Band_6-Day_3',
        'Unsorted MonthlyBands-Band_6-Day_7',
        'Unsorted MonthlyBands-Band_6-Day_8',
        'Unsorted MonthlyBands-Band_7-Day_3',
        'Unsorted MonthlyBands-Band_7-Day_6',
        'Unsorted MonthlyBands-Band_7-Day_7',
        'Unsorted MonthlyBands-Band_7-Day_8',
        'Unsorted MonthlyBands-Band_7-Day_9',
        'Unsorted MonthlyBands-NDVI-Day_1',
        'Lowest3MeanBandRefl-Band_6',
        'Lowest3MeanBandRefl-Band_8',
        'Lowest6MeanBandRefl-Band_1',
        'Lowest6MeanBandRefl-Band_7',
        'Lowest6MeanBandRefl-Band_8',
        'Lowest8MeanBandRefl-Band_1',
        'Lowest8MeanBandRefl-Band_2',
        'Lowest8MeanBandRefl-Band_7',
        'BandReflMedian-Band_2',
        'BandReflMedian-Band_8',
        # 'N/A-Band_1',
        # 'N/A-Band_2',
        'AmpBandRefl-Band_7',
        'AmpBandRefl-Band_8',
        # 'N/A-NDVI-Day_2',
        # 'N/A-NDVI-Day_3',
        'BandReflMaxGreenness-Band_2',
        'BandReflMaxGreenness-NDVI',
        'Greenest3MeanBandRefl-Band_1',
        'Greenest3MeanBandRefl-Band_8',
        'Greenest3MeanBandRefl-NDVI',
        'Greenest6MeanBandRefl-Band_1',
        'Greenest6MeanBandRefl-Band_8',
        'Greenest6MeanBandRefl-NDVI',
        'Greenest8MeanBandRefl-Band_2',
        'Greenest8MeanBandRefl-Band_7',
        'BandReflMedianGreenness-Band_6',
        'BandReflMedianGreenness-NDVI',
        'AmpGreenestBandRefl-Band_8',
        'Warmest3MeanBandRefl-Band_1',
        'Warmest3MeanBandRefl-Band_2',
        'Warmest3MeanBandRefl-Band_6',
        'Warmest3MeanBandRefl-Band_7',
        'Warmest3MeanBandRefl-Band_8',
        'Warmest3MeanBandRefl-NDVI',
        'Warmest6MeanBandRefl-Band_8',
        'Warmest6MeanBandRefl-NDVI',
        'Warmest8MeanBandRefl-Band_7',
        'Warmest8MeanBandRefl-Band_8',
        'Warmest8MeanBandRefl-NDVI']
    
    mcs = MonteCarloSim(trainingType=args.trainingType, args.o)
    finalModel: RandomForestRegressor = mcs.runFinalModel(pgePreds)
    mcs.saveFinalModel(finalModel)


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
