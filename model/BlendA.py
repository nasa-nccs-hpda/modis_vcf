from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# BlendA
#
# The v1.4.0 map steps after the two models, moved into the package (v1.4.1):
#
# Blend A (tree layer, MODIS tile rows v06-v11 = 30N-30S), from v1.4.0's
# blend_year.py and gate_common.py (blend/, removed in v1.5.2; at tag v1.5.1):
#   tree = floor((1 - w) * clear + w * partner + 0.5) on pixels valid (0-100)
#   in both, w = cloud weight x cover weight:
#   cloud weight = CUR_ZS / 8 where CUR_ZS >= 5 of the 8 periods, else 0
#   (CUR_ZS: periods whose composite fell back to an all-cloudy mean, with no
#   input of sun zenith 0-65, snow picks not counted; model/V4r.py);
#   cover weight = clip((clear - 20) / 30, 0, 1).
#   The partner (V4R60) is a tree model fitted on the V4r composites, which
#   are built without the low-sun substitution.
#
# BP10, from v1.4.0's bp10_apply.py (every tile): where the bare model's uncapped
# value is 100 and the final tree is 1-9, tree = 0, non-vegetated = 100,
# non-tree vegetation = 0.
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
    model/V4r.py in the metrics task for the year being mapped)."""

    f = Path(countsDir) / (tid + '_counts_snow.npz')

    if not f.exists():
        raise RuntimeError('Blend A needs the cloud counts ' + str(f) +
                           ' (model/V4r.py, metrics task, for this tile-year).')

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
def blend(clear: np.ndarray, partner: np.ndarray, counts: np.ndarray) -> tuple:
    """(blended tree, w, stats).  Pixels not valid in both keep clear."""

    ok = lambda v: (v >= 0) & (v <= 100)
    c = clear.astype(np.float64)
    valid = ok(c) & ok(partner)
    w = weights(counts, c)
    value = np.floor((1 - w) * c + w * partner.astype(np.float64) + 0.5)
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


# ----------------------------------------------------------------------------
# bp10
# ----------------------------------------------------------------------------
def bp10(tree: np.ndarray, bare: np.ndarray, nontree: np.ndarray,
         rawBare: np.ndarray) -> tuple:
    """(tree, bare, non-tree, pixels changed) after BP10."""

    ok = lambda v: (v >= 0) & (v <= 100)
    land = ok(tree) & ok(bare) & ok(nontree)
    bp = land & (rawBare == 100) & (tree >= 1) & (tree <= 9)
    tree, bare, nontree = tree.copy(), bare.copy(), nontree.copy()
    tree[bp] = 0
    bare[bp] = 100
    nontree[bp] = 0
    return tree, bare, nontree, int(bp.sum())
