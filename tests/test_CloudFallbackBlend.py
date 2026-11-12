"""model/CloudFallbackBlend.py: scope rows, weights and the cloud-fallback tree blend."""
import numpy as np

from modis_vcf.model import CloudFallbackBlend as B


def test_scope():
    assert [v for v in range(18) if B.inScope(v)] == [6, 7, 8, 9, 10, 11]


def test_weights():
    counts = np.array([4, 5, 8, 8, 8], np.int16)
    clear = np.array([60, 60, 60, 20, 35], np.int16)
    w = B.weights(counts, clear)
    # cloud weight 0 below 5 periods, CUR_ZS / 8 from 5; cover weight clip((clear - 20) / 30, 0, 1)
    assert np.allclose(w, [0, 5 / 8, 1, 0, 0.5])


def test_blend():
    clear = np.array([60, 60, 35, 200], np.int16)
    fallback = np.array([80, 20, 75, 200], np.int16)
    counts = np.array([8, 4, 8, 8], np.int16)
    tree, w, stats = B.blend(clear, fallback, counts)
    # floor((1 - w) * clear + w * fallback + 0.5): w = 1 -> 80; w = 0 -> 60; w = 0.5 -> 55; water kept
    assert tree.tolist() == [80, 60, 55, 200]
    assert stats['unchanged_where_w0']
