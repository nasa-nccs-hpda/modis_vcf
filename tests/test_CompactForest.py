"""model/CompactForest.py: the compact copy predicts as scikit-learn, tree by tree; save / load / memory-map."""
import numpy as np
import pytest
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

from modis_vcf.model.CompactForest import CompactForest
from modis_vcf.model.VcfPredict import VcfPredict


@pytest.fixture(scope='module')
def forest():
    rng = np.random.default_rng(0)
    X = rng.integers(-2000, 10000, (2000, 6)).astype(np.float32)
    y = np.clip(X[:, 0] / 100 + rng.normal(0, 5, 2000), 0, 100)
    return RandomForestRegressor(n_estimators=12, max_features='sqrt', random_state=0).fit(X, y), X


def test_trees_identical(forest):
    model, X = forest
    cf = CompactForest.fromSklearn(model)
    assert len(cf.estimators_) == 12 and cf.n_features_in_ == 6
    for sk, c in zip(model.estimators_, cf.estimators_):
        assert np.array_equal(sk.predict(X), c.predict(X))


def test_product_aggregation_identical(forest):
    model, X = forest
    cf = CompactForest.fromSklearn(model)
    for agg in ('median', 'mean'):
        a, sda = VcfPredict._memberPredict(model, X, agg)
        b, sdb = VcfPredict._memberPredict(cf, X, agg)
        assert np.array_equal(VcfPredict._roundValue(a), VcfPredict._roundValue(b)) and np.array_equal(sda, sdb)


def test_save_load_mmap(forest, tmp_path):
    model, X = forest
    CompactForest.fromSklearn(model, source='test').save(tmp_path / 'm.cf')
    assert CompactForest.isCompact(tmp_path / 'm.cf') and not CompactForest.isCompact(tmp_path)
    with pytest.raises(FileExistsError):
        CompactForest.fromSklearn(model).save(tmp_path / 'm.cf')
    for mmap in (False, True):
        cf = CompactForest.load(tmp_path / 'm.cf', mmap=mmap)
        assert cf.meta['n_trees'] == 12 and cf.meta['source'] == 'test'
        assert all(np.array_equal(s.predict(X), c.predict(X)) for s, c in zip(model.estimators_, cf.estimators_))


def test_refuses_bagging_with_feature_subsets():
    rng = np.random.default_rng(1)
    X, y = rng.random((200, 4)).astype(np.float32), rng.random(200)
    bag = BaggingRegressor(n_estimators=3, max_features=2, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        CompactForest.fromSklearn(bag)
