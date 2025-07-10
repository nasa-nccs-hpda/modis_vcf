#!/usr/bin/python

import argparse
import joblib
import multiprocessing
from pathlib import Path
import pickle
import sys

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from modis_vcf.model.TrainingType import TrainingType


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/rfFitFromPge61.py -o /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles-06-2025/MOD44C/pge61Model --xTrainPath /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles-06-2025/MOD44C/3-Models/pcttree/xTrain.parq --yTrainPath /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles-06-2025/MOD44C/3-Models/pcttree/ytrain.parq --trainingType pcttree
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to run a Monte Carlo simulation on metrics.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-o',
                        type=Path,
                        required='True',
                        help='Write the final model here.')

    parser.add_argument('--xTrainPath',
                        required=True,
                        type=Path,
                        help='Directory containing xTrain.parq')

    parser.add_argument('--yTrainPath',
                        required=True,
                        type=Path,
                        help='Directory containing yTrain.parq')

    parser.add_argument('--trainingType',
                        type=TrainingType,
                        choices=[t.value for t in TrainingType],
                        help='Choose the training type to run.')

    args = parser.parse_args()

    treeCoverPreds = [
        'UnsortedMonthlyBands-Band_1-Day_2019065',
        'UnsortedMonthlyBands-Band_2-Day_2019065',
        'UnsortedMonthlyBands-Band_3-Day_2019065',
        'UnsortedMonthlyBands-Band_4-Day_2019065',
        'UnsortedMonthlyBands-Band_5-Day_2019065',
        'UnsortedMonthlyBands-Band_6-Day_2019065',
        'UnsortedMonthlyBands-Band_7-Day_2019065',
        'UnsortedMonthlyBands-NDVI-Day_2019065',
        'UnsortedMonthlyBands-Band_1-Day_2019113',
        'UnsortedMonthlyBands-Band_2-Day_2019113',
        'UnsortedMonthlyBands-Band_4-Day_2019113',
        'UnsortedMonthlyBands-Band_6-Day_2019113',
        'UnsortedMonthlyBands-Band_7-Day_2019113',
        'UnsortedMonthlyBands-Band_7-Day_2019257',
        'UnsortedMonthlyBands-NDVI-Day_2019257',
        'UnsortedMonthlyBands-Band_3-Day_2019305',
        'UnsortedMonthlyBands-Band_7-Day_2019305',
        'UnsortedMonthlyBands-Band31-Day_2019305',
        'UnsortedMonthlyBands-Band_3-Day_2019353',
        'UnsortedMonthlyBands-Band_6-Day_2019353',
        'UnsortedMonthlyBands-Band_7-Day_2019353',
        'UnsortedMonthlyBands-Band31-Day_2019353',
        'UnsortedMonthlyBands-NDVI-Day_2019353',
        'UnsortedMonthlyBands-Band_1-Day_2020033',
        'Lowest3MeanBandRefl-Band_6',
        'Lowest3MeanBandRefl-Band31',
        'Lowest6MeanBandRefl-Band_1',
        'Lowest6MeanBandRefl-Band_7',
        'Lowest6MeanBandRefl-Band31',
        'Lowest8MeanBandRefl-Band_1',
        'Lowest8MeanBandRefl-Band_2',
        'Lowest8MeanBandRefl-Band_7',
        'BandReflMedian-Band_2',
        'BandReflMedian-Band31',
        # 'N/A-Band_1',
        # 'N/A-Band_2',
        'AmpBandRefl-Band_7',
        'AmpBandRefl-Band31',
        # 'N/A-Band_2-Day_2019113',
        # 'N/A-Band_3-Day_2019161',
        'BandReflMaxGreenness-Band_2',
        'BandReflMaxGreenness-NDVI',
        'Greenest3MeanBandRefl-Band_1',
        'Greenest3MeanBandRefl-Band31',
        'Greenest3MeanBandRefl-NDVI',
        'Greenest6MeanBandRefl-Band_1',
        'Greenest6MeanBandRefl-Band31',
        'Greenest6MeanBandRefl-NDVI',
        'Greenest8MeanBandRefl-Band_2',
        'Greenest8MeanBandRefl-Band_7',
        'BandReflMedianGreenness-Band_6',
        'BandReflMedianGreenness-NDVI',
        'AmpGreenestBandRefl-Band31',
        'Warmest3MeanBandRefl-Band_1',
        'Warmest3MeanBandRefl-Band_2',
        'Warmest3MeanBandRefl-Band_6',
        'Warmest3MeanBandRefl-Band_7',
        'Warmest3MeanBandRefl-Band31',
        'Warmest3MeanBandRefl-NDVI',
        'Warmest6MeanBandRefl-Band31',
        'Warmest6MeanBandRefl-NDVI',
        'Warmest8MeanBandRefl-Band_7',
        'Warmest8MeanBandRefl-Band31',
        'Warmest8MeanBandRefl-NDVI']
        
    barePreds = [
    'UnsortedMonthlyBands-Band_6-Day_2019113',
    'UnsortedMonthlyBands-Band_1-Day_2019161',
    'UnsortedMonthlyBands-NDVI-Day_2019305',
    'UnsortedMonthlyBands-Band_3-Day_2019353',
    'UnsortedMonthlyBands-Band_7-Day_2019353',
    'UnsortedMonthlyBands-Band31-Day_2019353',
    'UnsortedMonthlyBands-NDVI-Day_2019353',
    'UnsortedMonthlyBands-Band_1-Day_2020033',
    'UnsortedMonthlyBands-Band_2-Day_2020033',
    'UnsortedMonthlyBands-Band_3-Day_2020033',
    'UnsortedMonthlyBands-Band_4-Day_2020033',
    'UnsortedMonthlyBands-Band_5-Day_2020033',
    'UnsortedMonthlyBands-Band_7-Day_2020033',
    'UnsortedMonthlyBands-Band31-Day_2020033',
    'UnsortedMonthlyBands-NDVI-Day_2020033',
    'Lowest6MeanBandRefl-Band_4',
    'BandReflMedian-NDVI',
    # 'N/A-Band_2',
    # 'N/A-Band_6-Day_2019305',
    # 'BandReflMin-Band_1-Day_2019065',
    'Greenest6MeanBandRefl-Band31']

    outDir = args.o
    trainingType = args.trainingType
    
    preds = treeCoverPreds if trainingType == TrainingType.PCT_TREE \
        else barePreds
    
    print('Reading training data.')
    xTrain = pd.read_parquet(args.xTrainPath, columns=preds)
    yTrain = pd.read_parquet(args.yTrainPath).to_numpy().ravel()

    print('Fitting pcttree model.')
    rf = RandomForestRegressor(n_estimators=1)
    rf = rf.fit(xTrain, yTrain)

    outPath: Path = outDir / (trainingType + '.bin')
    topPath: Path = outDir / (trainingType.value + 'TopN.bin')
    
    with open(outPath, 'wb') as f:
        joblib.dump(rf, f)

    with open(topPath, 'wb') as f:
        pickle.dump(preds, f)


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
