#!/usr/bin/env python3
"""Fit the Random Forest model (scikit-learn RandomForestRegressor) of one layer from its xTrain / yTrain split.

Takes the predictor list and the split, and writes <trainingType>.bin (the fitted model, joblib) and <trainingType>TopN.bin
(the predictor names, pickle), which the predict path (vcfPredict.py -> VcfPredict.py) reads.  YearRun runs this for every
fitted layer, then makes the compact copy with CompactForest.

Usage (tree cover; non-vegetated is the same with --trainingType pctbare and its own split):
  fitAltModel.py --algo rf -o <outDir>/rf --xTrainPath <training>/pcttree/xTrain.parq \
    --yTrainPath <training>/pcttree/yTrain.parq --trainingType pcttree \
    --predictorFile <predictors_pcttree.txt> --nEstimators 60 --nJobs 12 --randomState 0
"""
import argparse
import joblib
from pathlib import Path
import pickle
import sys
import time

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--algo', required=True, choices=['rf'], help='rf (the only model; kept so existing command lines still work).')
    ap.add_argument('-o', type=Path, required=True, help='Write the fitted model here.')
    ap.add_argument('--xTrainPath', type=Path, required=True)
    ap.add_argument('--yTrainPath', type=Path, required=True)
    ap.add_argument('--trainingType', required=True, help='Used only to name the output files (pcttree.bin, pcttreeTopN.bin).')
    ap.add_argument('--predictorFile', type=Path, required=True, help='One predictor column name per line.')
    ap.add_argument('--nJobs', type=int, default=1)
    ap.add_argument('--randomState', type=int, default=None)

    ap.add_argument('--nEstimators', type=int, default=30, help='Number of trees (default 30; the product uses 60).')
    ap.add_argument('--minSamplesLeaf', type=int, default=1, help='min_samples_leaf (default 1 = sklearn default, as the product models).')
    ap.add_argument('--maxFeatures', default='sqrt', help='Features considered per split (sklearn default is 1.0; textbook RF regression default is 1/3 of features, sqrt is a common alternative).')

    args = ap.parse_args()

    preds = [l.strip() for l in open(args.predictorFile) if l.strip()]
    print('Using', len(preds), 'predictors from', args.predictorFile)

    print('Reading training data.')
    xTrain = pd.read_parquet(args.xTrainPath, columns=preds)
    yTrain = pd.read_parquet(args.yTrainPath).to_numpy().ravel()
    print('xTrain shape:', xTrain.shape)

    t0 = time.time()
    from sklearn.ensemble import RandomForestRegressor
    print('Fitting RandomForestRegressor: n_estimators=%d max_features=%s min_samples_leaf=%d n_jobs=%d random_state=%s'
          % (args.nEstimators, args.maxFeatures, args.minSamplesLeaf, args.nJobs, args.randomState))
    model = RandomForestRegressor(n_estimators=args.nEstimators,
                                  max_features=args.maxFeatures,
                                  min_samples_leaf=args.minSamplesLeaf,
                                  n_jobs=args.nJobs,
                                  random_state=args.randomState)

    model.fit(xTrain, yTrain)
    print('Fit took %.1f minutes.' % ((time.time() - t0) / 60.0))

    args.o.mkdir(parents=True, exist_ok=True)
    outPath = args.o / (args.trainingType + '.bin')
    topPath = args.o / (args.trainingType + 'TopN.bin')

    with open(outPath, 'wb') as f:
        joblib.dump(model, f)
    with open(topPath, 'wb') as f:
        pickle.dump(preds, f)

    print('Wrote', outPath, 'and', topPath)


if __name__ == '__main__':
    sys.exit(main())
