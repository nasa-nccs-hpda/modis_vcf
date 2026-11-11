import unittest

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor

from modis_vcf.model.VcfPredict import VcfPredict


# -----------------------------------------------------------------------------
# class VcfPredictMembersTestCase
#
# Checks the ensemble-member pass (value aggregation and SD) on small models fitted to synthetic data, so it needs
# no tiles, metrics or model files.
#
# python -m unittest modis_vcf.model.tests.test_VcfPredict_members
# -----------------------------------------------------------------------------
class VcfPredictMembersTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUp
    # -------------------------------------------------------------------------
    def setUp(self):

        rng = np.random.default_rng(0)
        self._X = rng.uniform(0, 1000, size=(3000, 6)).astype(np.float32)
        y = np.clip(self._X[:, 0] / 10 + rng.normal(0, 15, len(self._X)), 0, 100)
        xTrain, yTrain = self._X[:2000], y[:2000]
        self._XTest = self._X[2000:]

        self._rf = RandomForestRegressor(n_estimators=30, max_features='sqrt', random_state=0).fit(xTrain, yTrain)

        self._bag = BaggingRegressor(n_estimators=30, max_features=4, random_state=0).fit(xTrain, yTrain)

    # -------------------------------------------------------------------------
    # memberPreds
    # -------------------------------------------------------------------------
    @staticmethod
    def _memberPreds(model, X):

        return np.vstack([np.clip(est.predict(X if feats is None else X[:, feats]), 0, 100)
                          for est, feats in VcfPredict._members(model)])

    # -------------------------------------------------------------------------
    # testMeanEqualsPredict
    # -------------------------------------------------------------------------
    def testMeanEqualsPredict(self):

        for model in (self._rf, self._bag):

            value, _ = VcfPredict._memberPredict(model, self._XTest, 'mean', nJobs=2)
            self.assertTrue(np.allclose(value, model.predict(self._XTest), rtol=0, atol=1e-9))

    # -------------------------------------------------------------------------
    # testAggregations
    # -------------------------------------------------------------------------
    def testAggregations(self):

        for model in (self._rf, self._bag):

            P = self._memberPreds(model, self._XTest)
            S = np.sort(P, axis=0)

            median, _ = VcfPredict._memberPredict(model, self._XTest, 'median', nJobs=2)
            np.testing.assert_array_equal(median, np.median(P, axis=0))

            mid20, _ = VcfPredict._memberPredict(model, self._XTest, 'mid20', nJobs=2)
            np.testing.assert_allclose(mid20, S[5:25].mean(axis=0), rtol=0, atol=1e-12)

    # -------------------------------------------------------------------------
    # testSd
    # -------------------------------------------------------------------------
    def testSd(self):

        for model in (self._rf, self._bag):

            P = self._memberPreds(model, self._XTest)

            for agg in VcfPredict.AGG_METHODS:

                _, sd = VcfPredict._memberPredict(model, self._XTest, agg, nJobs=2)
                np.testing.assert_allclose(sd, P.std(axis=0), rtol=0, atol=1e-12)

    # -------------------------------------------------------------------------
    # testChunking
    # -------------------------------------------------------------------------
    def testChunking(self):

        saved = VcfPredict.PREDICT_CHUNK_ROWS

        try:
            whole = VcfPredict._memberPredict(self._rf, self._XTest, 'median')
            VcfPredict.PREDICT_CHUNK_ROWS = 333
            chunked = VcfPredict._memberPredict(self._rf, self._XTest, 'median', nJobs=3)

        finally:
            VcfPredict.PREDICT_CHUNK_ROWS = saved

        np.testing.assert_array_equal(whole[0], chunked[0])
        np.testing.assert_array_equal(whole[1], chunked[1])

    # -------------------------------------------------------------------------
    # testNoMembers
    # -------------------------------------------------------------------------
    def testNoMembers(self):

        class NoMembers(object):
            pass

        self.assertIsNone(VcfPredict._members(NoMembers()))

        with self.assertRaises(ValueError):
            VcfPredict._combine(np.zeros((10, 5)), 'mid20')

        with self.assertRaises(ValueError):
            VcfPredict._combine(np.zeros((30, 5)), 'trimmed')
