import json
import sys
from pathlib import Path

import joblib
import numba
import numpy as np


# ----------------------------------------------------------------------------
# Class CompactForest
#
# The same trees as a fitted scikit-learn RandomForestRegressor, stored as flat arrays (about 10 bytes per node instead of
# scikit-learn's 64) and predicted by a small numba loop.  Predictions are identical to scikit-learn's, tree by tree:
#   - scikit-learn predicts on float32 X and goes left when x <= threshold (float64).  For any float32 x that is the same
#     test as x <= t32, where t32 is the largest float32 not above the threshold, which is what is stored.
#   - Leaf values are kept in float64 (a table of the distinct leaf values).
#   - Combining the trees (clip, median, SD, rounding) stays in VcfPredict, unchanged.
# A model is a directory (by convention <name>.cf) of .npy files plus meta.json, so it can be memory-mapped.
# Reports/Smaller_RF_Models_Compact_And_Subsample_Plan.md, part A (2026-10-14).
# ----------------------------------------------------------------------------
class CompactForest(object):

    FILES = ('feature', 'threshold', 'node', 'roots', 'values')

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, arrays: dict, meta: dict):

        self.feature: np.ndarray = arrays['feature']      # int16, -1 = leaf
        self.threshold: np.ndarray = arrays['threshold']  # float32
        self.node: np.ndarray = arrays['node']            # int32: right child (tree-relative) or leaf-value index
        self.roots: np.ndarray = arrays['roots']          # int64, n_trees + 1 offsets
        self.values: np.ndarray = arrays['values']        # float64, distinct leaf values
        self.meta: dict = meta
        self.n_features_in_: int = meta['n_features_in']

        self.estimators_ = [CompactTree(self, k) for k in range(len(self.roots) - 1)]

    # ------------------------------------------------------------------------
    # fromSklearn
    # ------------------------------------------------------------------------
    @staticmethod
    def fromSklearn(model, source: str = '') -> 'CompactForest':

        if getattr(model, 'estimators_features_', None) is not None:
            raise ValueError('Only RandomForestRegressor is supported (this model has per-member feature subsets).')

        if getattr(model, 'n_outputs_', 1) != 1:
            raise ValueError('Only single-output models are supported.')

        if model.n_features_in_ > np.iinfo(np.int16).max:
            raise ValueError('Too many features for int16.')

        feats, thrs, nodes, leafVals, roots = [], [], [], [], [0]

        for est in model.estimators_:

            t = est.tree_
            n = t.node_count
            left, right = t.children_left, t.children_right
            leaf = left == -1
            internal = ~leaf

            # The traversal relies on left child = node + 1 (scikit-learn's depth-first builder).
            if not np.array_equal(left[internal], np.flatnonzero(internal) + 1):
                raise ValueError('Left child is not node + 1; this tree layout is not supported.')

            if n > np.iinfo(np.int32).max:
                raise ValueError('Tree too large for int32 node indices.')

            thr = t.threshold.astype(np.float64)
            t32 = thr.astype(np.float32)
            up = t32.astype(np.float64) > thr
            t32[up] = np.nextafter(t32[up], np.float32(-np.inf))

            feats.append(np.where(leaf, -1, t.feature).astype(np.int16))
            thrs.append(np.where(leaf, np.float32(0), t32).astype(np.float32))
            nodes.append(np.where(leaf, 0, right).astype(np.int64))   # leaf slots filled below
            leafVals.append(t.value[:, 0, 0].astype(np.float64)[leaf])
            roots.append(roots[-1] + n)

        values = np.unique(np.concatenate(leafVals))

        if len(values) > np.iinfo(np.int32).max:
            raise ValueError('Too many distinct leaf values for int32.')

        for k in range(len(nodes)):
            leaf = feats[k] == -1
            nodes[k][leaf] = np.searchsorted(values, leafVals[k])
            nodes[k] = nodes[k].astype(np.int32)

        arrays = dict(feature=np.concatenate(feats), threshold=np.concatenate(thrs), node=np.concatenate(nodes),
                      roots=np.array(roots, dtype=np.int64), values=values)
        meta = dict(format='CompactForest 1', source=source, n_trees=len(model.estimators_),
                    n_features_in=int(model.n_features_in_), n_nodes=int(roots[-1]), n_values=int(len(values)),
                    sklearn_params={k: repr(v) for k, v in model.get_params().items()})

        return CompactForest(arrays, meta)

    # ------------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------------
    def save(self, outDir: Path) -> None:

        outDir = Path(outDir)
        outDir.mkdir(parents=True, exist_ok=False)

        for name in CompactForest.FILES:
            np.save(outDir / (name + '.npy'), getattr(self, name))

        (outDir / 'meta.json').write_text(json.dumps(self.meta, indent=1))

    # ------------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------------
    @staticmethod
    def load(inDir: Path, mmap: bool = False) -> 'CompactForest':

        inDir = Path(inDir)
        arrays = {name: np.load(inDir / (name + '.npy'), mmap_mode='r' if mmap else None) for name in CompactForest.FILES}
        meta = json.loads((inDir / 'meta.json').read_text())

        return CompactForest(arrays, meta)

    # ------------------------------------------------------------------------
    # isCompact
    # ------------------------------------------------------------------------
    @staticmethod
    def isCompact(path: Path) -> bool:
        return Path(path).is_dir() and (Path(path) / 'meta.json').exists()

    # ------------------------------------------------------------------------
    # set_params: n_jobs has no meaning here (VcfPredict threads the members); kept for VcfPredict's call.
    # ------------------------------------------------------------------------
    def set_params(self, **kwargs) -> 'CompactForest':
        return self

    # ------------------------------------------------------------------------
    # predict: the trees' mean, summed in tree order then divided (scikit-learn sums in thread-completion order,
    # so the two can differ in the last bit; the product uses the members' median, not this).
    # ------------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:

        X = CompactTree.prepare(X, self.n_features_in_)
        out = np.zeros(X.shape[0], dtype=np.float64)

        for est in self.estimators_:
            out += est.predict(X)

        return out / len(self.estimators_)


# ----------------------------------------------------------------------------
# Class CompactTree: one member, with a predict() that matches DecisionTreeRegressor.predict.
# ----------------------------------------------------------------------------
class CompactTree(object):

    def __init__(self, forest: CompactForest, k: int):

        a, b = int(forest.roots[k]), int(forest.roots[k + 1])
        self.feature = forest.feature[a:b]
        self.threshold = forest.threshold[a:b]
        self.node = forest.node[a:b]
        self.values = forest.values
        self.n_features_in_ = forest.n_features_in_

    @staticmethod
    def prepare(X, nFeatures: int) -> np.ndarray:
        """As scikit-learn's check_array for trees: float32, finite values only, the right number of columns."""
        X = np.ascontiguousarray(np.asarray(X), dtype=np.float32)

        if X.ndim != 2 or X.shape[1] != nFeatures:
            raise ValueError('X has shape %s; the model expects %d features.' % (X.shape, nFeatures))

        if not np.isfinite(X).all():
            raise ValueError('Input contains NaN or infinity.')

        return X

    def predict(self, X) -> np.ndarray:

        X = CompactTree.prepare(X, self.n_features_in_)
        out = np.empty(X.shape[0], dtype=np.float64)
        _walk(X, self.feature, self.threshold, self.node, self.values, out)

        return out


# ----------------------------------------------------------------------------
# _walk: one tree over the rows of X (releases the GIL, so VcfPredict's member threads run in parallel).
# ----------------------------------------------------------------------------
@numba.njit(nogil=True, cache=False)
def _walk(X, feature, threshold, node, values, out):

    for i in range(X.shape[0]):

        k = 0
        f = feature[0]

        while f >= 0:

            if X[i, f] <= threshold[k]:
                k += 1
            else:
                k = node[k]

            f = feature[k]

        out[i] = values[node[k]]


# ----------------------------------------------------------------------------
# Convert a joblib model: python -m modis_vcf.model.CompactForest <model.bin> <out dir (e.g. pcttree.cf)>
# ----------------------------------------------------------------------------
if __name__ == '__main__':

    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    CompactForest.fromSklearn(joblib.load(src), source=str(src)).save(dst)
    print('wrote', dst)
