from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# CloudFallbackBlend
#
# The cloud-fallback tree blend, a v1.4.0 map step after the two models, moved into the package (v1.4.1):
#
# Cloud-fallback tree blend (tree layer, MODIS tile rows v06-v11 = 30N-30S), from v1.4.0's
# blend_year.py and gate_common.py (blend/, removed in v1.5.2; at tag v1.5.1):
#   tree = floor((1 - w) * clear + w * fallbackTree + 0.5) on pixels valid (0-100)
#   in both, w = cloud weight x cover weight:
#   cloud weight = CUR_ZS / 8 where CUR_ZS >= 5 of the 8 periods, else 0
#   (CUR_ZS: periods whose composite fell back to an all-cloudy mean, with no
#   input of sun zenith 0-65, snow picks not counted; model/CloudFallback.py);
#   cover weight = clip((clear - 20) / 30, 0, 1).
#   The cloud-fallback tree model is a tree model fitted on the cloud-fallback composites, which
#   are built without the low-sun substitution.
#
# The bare-priority rule (BP10 in v1.4.0), the last step of the map, is in
# model/BarePriority.py.
# ----------------------------------------------------------------------------
SCOPE_VTILES = tuple(range(6, 12))
CLOUD_MIN_PERIODS = 5
PERIODS = 8
COVER_LOW = 20
COVER_SPAN = 30
COUNTS_KEY = 'CUR_ZS'


# ----------------------------------------------------------------------------
# inScope
# ----------------------------------------------------------------------------
def inScope(vTile: int) -> bool:
    return vTile in SCOPE_VTILES


# ----------------------------------------------------------------------------
# loadCounts
# ----------------------------------------------------------------------------
def loadCounts(countsDir: Path, tid: str) -> np.ndarray:
    """CUR_ZS from <countsDir>/<tile>_counts_snow.npz (written by
    model/CloudFallback.py in the metrics task for the year being mapped)."""

    f = Path(countsDir) / (tid + '_counts_snow.npz')

    if not f.exists():
        raise RuntimeError('Cloud-fallback tree blend needs the cloud counts ' + str(f) +
                           ' (model/CloudFallback.py, metrics task, for this tile-year).')

    return np.load(f)[COUNTS_KEY]


# ----------------------------------------------------------------------------
# weights
# ----------------------------------------------------------------------------
def weights(counts: np.ndarray, clear: np.ndarray) -> np.ndarray:

    c = counts.astype(np.float64)
    cloud = np.where(c >= CLOUD_MIN_PERIODS, c / PERIODS, 0.0)
    cover = np.clip((clear.astype(np.float64) - COVER_LOW) / COVER_SPAN, 0, 1)
    return cloud * cover


# ----------------------------------------------------------------------------
# blend
# ----------------------------------------------------------------------------
def blend(clear: np.ndarray, fallbackTree: np.ndarray, counts: np.ndarray) -> tuple:
    """(blended tree, w, stats).  Pixels not valid in both keep clear."""

    ok = lambda v: (v >= 0) & (v <= 100)
    c = clear.astype(np.float64)
    valid = ok(c) & ok(fallbackTree)
    w = weights(counts, c)
    value = np.floor((1 - w) * c + w * fallbackTree.astype(np.float64) + 0.5)
    tree = clear.copy()
    tree[valid] = value[valid].astype(clear.dtype)

    land = ok(clear)
    changed = land & (tree != clear)
    stats = dict(land=int(land.sum()), w_pos=int((land & (w > 0)).sum()),
                 tree_changed=int(changed.sum()),
                 tree_up=int((land & (tree > clear)).sum()),
                 tree_down=int((land & (tree < clear)).sum()),
                 unchanged_where_w0=bool((tree[land & (w == 0)] ==
                                          clear[land & (w == 0)]).all()))
    return tree, w, stats

