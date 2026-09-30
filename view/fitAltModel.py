#!/usr/bin/env python3
"""Fit a Random Forest, XGBoost, or Histogram Gradient Boosting model, as an alternative to the
BaggingRegressor(DecisionTreeRegressor()) model from rfFitFromPge61.py. It takes the same predictors
and xTrain/yTrain, and writes the same output files (<trainingType>.bin and <trainingType>TopN.bin).

RF and XGBoost models run through the existing predict path (vcfPredictQueue.py -> VcfPredict.py)
unchanged: it only calls joblib.load, .set_params(n_jobs=...) and .predict(). XGBoost can predict
outside 0-100; VcfPredict clamps before applying the water/out-of-projection codes.
HistGradientBoostingRegressor has no n_jobs parameter, so VcfPredict's .set_params(n_jobs=...) would
fail on an hgb model; hgb is for held-out scoring only.

In the 2026 MODIS-VCF evaluation (docs/reports/Regression_Tree_Methods_RF_XGBoost_Results.md), XGBoost
"tunedB" (below) was the most accurate of the three Python models against the historical MOD44B.

Usage (tree cover; nonveg is the same with --trainingType pctbare and its own training data):
  fitAltModel.py --algo rf -o <outDir>/rf --xTrainPath <training>/pcttree/xTrain.parq \
    --yTrainPath <training>/pcttree/yTrain.parq --trainingType pcttree \
    --predictorFile <predictors_pcttree.txt> --nEstimators 30 --nJobs 12 --randomState 0

  # XGBoost tunedB
  fitAltModel.py --algo xgboost -o <outDir>/xgboost_tunedB --xTrainPath ... --yTrainPath ... \
    --trainingType pcttree --predictorFile ... --nEstimators 2000 --maxDepth 6 --learningRate 0.02 \
    --subsample 0.8 --colsampleBytree 0.8 --nJobs 12 --randomState 0

  fitAltModel.py --algo hgb -o <outDir>/hgb --xTrainPath ... --yTrainPath ... \
    --trainingType pcttree --predictorFile ... --nEstimators 2000 --maxDepth 6 --learningRate 0.02 \
    --randomState 0   # subsample/colsampleBytree are ignored for hgb
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
    ap.add_argument('--algo', required=True, choices=['rf', 'xgboost', 'hgb'])
    ap.add_argument('-o', type=Path, required=True, help='Write the fitted model here.')
    ap.add_argument('--xTrainPath', type=Path, required=True)
    ap.add_argument('--yTrainPath', type=Path, required=True)
    ap.add_argument('--trainingType', required=True, help='Used only to name the output files (pcttree.bin, pcttreeTopN.bin).')
    ap.add_argument('--predictorFile', type=Path, required=True, help='One predictor column name per line.')
    ap.add_argument('--nJobs', type=int, default=1)
    ap.add_argument('--randomState', type=int, default=None)

    # Random Forest knobs
    ap.add_argument('--nEstimators', type=int, default=30, help='RF: number of trees (default 30, matches the baseline ensemble size). XGBoost: number of boosting rounds (default 500 if unset).')
    ap.add_argument('--maxFeatures', default='sqrt', help='RF only: features considered per split (sklearn default is 1.0; textbook RF regression default is 1/3 of features, sqrt is a common alternative).')

    # XGBoost knobs
    ap.add_argument('--maxDepth', type=int, default=6)
    ap.add_argument('--learningRate', type=float, default=0.05)
    ap.add_argument('--subsample', type=float, default=0.8)
    ap.add_argument('--colsampleBytree', type=float, default=0.8)

    args = ap.parse_args()

    preds = [l.strip() for l in open(args.predictorFile) if l.strip()]
    print('Using', len(preds), 'predictors from', args.predictorFile)

    print('Reading training data.')
    xTrain = pd.read_parquet(args.xTrainPath, columns=preds)
    yTrain = pd.read_parquet(args.yTrainPath).to_numpy().ravel()
    print('xTrain shape:', xTrain.shape)

    t0 = time.time()
    if args.algo == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        print('Fitting RandomForestRegressor: n_estimators=%d max_features=%s n_jobs=%d random_state=%s'
              % (args.nEstimators, args.maxFeatures, args.nJobs, args.randomState))
        model = RandomForestRegressor(n_estimators=args.nEstimators,
                                       max_features=args.maxFeatures,
                                       n_jobs=args.nJobs,
                                       random_state=args.randomState)
    elif args.algo == 'hgb':
        from sklearn.ensemble import HistGradientBoostingRegressor
        # No n_jobs (parallelizes internally) and no subsample/colsample_bytree (sklearn's HGB has
        # no stochastic row/column sub-sampling as of 1.2); --subsample/--colsampleBytree are
        # accepted on the command line but not used here, so a launcher can pass the same
        # tunedA/tunedB-style arguments uniformly across algorithms without erroring.
        print('Fitting HistGradientBoostingRegressor: max_iter=%d max_depth=%d learning_rate=%s random_state=%s'
              % (args.nEstimators, args.maxDepth, args.learningRate, args.randomState))
        model = HistGradientBoostingRegressor(max_iter=args.nEstimators,
                                               max_depth=args.maxDepth,
                                               learning_rate=args.learningRate,
                                               random_state=args.randomState)
    else:
        from xgboost import XGBRegressor
        # nEstimators default of 30 (the argparse default, matched to the baseline) is almost
        # certainly too few boosting rounds; the launcher should pass --nEstimators 500 explicitly
        # for xgboost, per the plan. Left as an explicit CLI value (not silently overridden here)
        # so a run's actual settings are always visible in its own command line / log.
        print('Fitting XGBRegressor: n_estimators=%d max_depth=%d learning_rate=%s subsample=%s '
              'colsample_bytree=%s n_jobs=%d random_state=%s'
              % (args.nEstimators, args.maxDepth, args.learningRate, args.subsample,
                 args.colsampleBytree, args.nJobs, args.randomState))
        model = XGBRegressor(n_estimators=args.nEstimators,
                              max_depth=args.maxDepth,
                              learning_rate=args.learningRate,
                              subsample=args.subsample,
                              colsample_bytree=args.colsampleBytree,
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
