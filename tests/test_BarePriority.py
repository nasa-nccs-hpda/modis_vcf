"""model/BarePriority.py: the bare-priority rule (BP10 up to v1.8.0)."""
import numpy as np

from modis_vcf.model import BarePriority


def test_rule():
    # columns: hit (raw 100, tree 5); tree 10 kept; raw 99 kept; tree 0 already; water; outside projection
    tree = np.array([[5, 10, 5, 0, 200, 250]], np.int16)
    bare = np.array([[90, 90, 90, 100, 200, 250]], np.int16)
    ntv = np.array([[5, 0, 5, 0, 200, 250]], np.int16)
    raw = np.array([[100, 100, 99, 100, 200, 250]], np.int16)
    t, b, n, count = BarePriority.apply(tree, bare, ntv, raw)
    assert count == 1
    assert t.tolist() == [[0, 10, 5, 0, 200, 250]]
    assert b.tolist() == [[100, 90, 90, 100, 200, 250]]
    assert n.tolist() == [[0, 0, 5, 0, 200, 250]]
    assert tree[0, 0] == 5                                   # inputs not modified


def test_tree_range_edges():
    tree = np.array([[1, 9, 10]], np.int16)
    raw = np.full((1, 3), 100, np.int16)
    t, b, n, count = BarePriority.apply(tree, np.full((1, 3), 90, np.int16), np.zeros((1, 3), np.int16), raw)
    assert count == 2 and t.tolist() == [[0, 0, 10]]
