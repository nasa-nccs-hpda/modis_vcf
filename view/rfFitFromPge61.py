#!/usr/bin/python

import argparse
import joblib
import multiprocessing
from pathlib import Path
import pickle
import sys

import pandas as pd

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

from modis_vcf.model import CAttributes
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

    parser.add_argument('--nEstimators',
                        type=int,
                        default=1,
                        help='Number of BaggingRegressor ensemble members. ' +
                             'Use 1 to fit a single per-model-number ' +
                             'member (mirrors C\'s per-model-number ' +
                             'weka.bare.NN.model structure); use 30 to fit ' +
                             'one complete ensemble matching C\'s 30-model ' +
                             'combined ensemble in a single invocation.')

    parser.add_argument('--predictorFile',
                        type=Path,
                        help='Text file with one predictor name per line; replaces the built-in list.')

    parser.add_argument('--nJobs',
                        type=int,
                        default=1,
                        help='Cores used to fit the ensemble members.')

    parser.add_argument('--randomState',
                        type=int,
                        default=None,
                        help='Random seed; with a seed the result does not depend on nJobs.')

    parser.add_argument('--year',
                        type=int,
                        default=2019,
                        help='Year of the training metrics; sets the period ' +
                             'days in the built-in non-vegetated predictor ' +
                             'names.')

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
        'UnsortedMonthlyBands-Band_31-Day_2019305',
        'UnsortedMonthlyBands-Band_3-Day_2019353',
        'UnsortedMonthlyBands-Band_6-Day_2019353',
        'UnsortedMonthlyBands-Band_7-Day_2019353',
        'UnsortedMonthlyBands-Band_31-Day_2019353',
        'UnsortedMonthlyBands-NDVI-Day_2019353',
        'UnsortedMonthlyBands-Band_1-Day_2020033',
        'Lowest3MeanBandRefl-Band_6',
        'Lowest3MeanBandRefl-Band_31',
        'Lowest6MeanBandRefl-Band_1',
        'Lowest6MeanBandRefl-Band_7',
        'Lowest6MeanBandRefl-Band_31',
        'Lowest8MeanBandRefl-Band_1',
        'Lowest8MeanBandRefl-Band_2',
        'Lowest8MeanBandRefl-Band_7',
        'BandReflMedian-Band_2',
        'BandReflMedian-Band_31',
        # 'N/A-Band_1',
        # 'N/A-Band_2',
        'AmpBandRefl-Band_7',
        'AmpBandRefl-Band_31',
        # 'N/A-Band_2-Day_2019113',
        # 'N/A-Band_3-Day_2019161',
        'BandReflMaxGreenness-Band_2',
        'BandReflMaxGreenness-NDVI',
        'Greenest3MeanBandRefl-Band_1',
        'Greenest3MeanBandRefl-Band_31',
        'Greenest3MeanBandRefl-NDVI',
        'Greenest6MeanBandRefl-Band_1',
        'Greenest6MeanBandRefl-Band_31',
        'Greenest6MeanBandRefl-NDVI',
        'Greenest8MeanBandRefl-Band_2',
        'Greenest8MeanBandRefl-Band_7',
        'BandReflMedianGreenness-Band_6',
        'BandReflMedianGreenness-NDVI',
        'AmpGreenestBandRefl-Band_31',
        'Warmest3MeanBandRefl-Band_1',
        'Warmest3MeanBandRefl-Band_2',
        'Warmest3MeanBandRefl-Band_6',
        'Warmest3MeanBandRefl-Band_7',
        'Warmest3MeanBandRefl-Band_31',
        'Warmest3MeanBandRefl-NDVI',
        'Warmest6MeanBandRefl-Band_31',
        'Warmest6MeanBandRefl-NDVI',
        'Warmest8MeanBandRefl-Band_7',
        'Warmest8MeanBandRefl-Band_31',
        'Warmest8MeanBandRefl-NDVI']
        
    # ---
    # The non-vegetated predictors are C's 21 bare attributes
    # (MOD_PR44B3/MOD44B3.c attributes_bare[], header.bare.arff), named as
    # the Python metrics that mirror them.  The former hand-written list
    # translated C's indices with a date-major layout instead of C's
    # band-major one, so 13 of its 18 names were not C's predictors.
    # ---
    barePreds = CAttributes.predictorNames(CAttributes.BARE_ATTRIBUTES,
                                           CAttributes.periodDays(args.year))

    outDir = args.o
    trainingType = args.trainingType
    
    preds = treeCoverPreds if trainingType == TrainingType.PCT_TREE \
        else barePreds

    if args.predictorFile:

        preds = [l.strip() for l in open(args.predictorFile) if l.strip()]
        print('Using', len(preds), 'predictors from', args.predictorFile)
    
    print('Reading training data.')
    xTrain = pd.read_parquet(args.xTrainPath, columns=preds)
    yTrain = pd.read_parquet(args.yTrainPath).to_numpy().ravel()

    print('Fitting pcttree model.')

    # ---
    # C's training method is Weka's Bagging meta-classifier around a single
    # tree learner (bootstrap-resamples rows, all features per split), not
    # Random Forest, which also restricts each split to a random feature
    # subset. BaggingRegressor wrapping a plain DecisionTreeRegressor is the
    # closer sklearn analogue; see
    # Sync_Plan_and_Model_Retraining_Alternate.md. n_estimators=1 matches
    # this script fitting one ensemble member per invocation, mirroring
    # C's per-model-number (weka.bare.NN.model) training structure.
    # ---
    rf = BaggingRegressor(estimator=DecisionTreeRegressor(),
                          n_estimators=args.nEstimators,
                          bootstrap=True,
                          n_jobs=args.nJobs,
                          random_state=args.randomState)
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
