"""Bare-priority rule: the last step of the map (product rule since v1.4.0, where it was called BP10).

Where the non-vegetated (bare) model is fully confident (its raw, uncapped value is 100) and the final tree cover is low
(1-9 %), the tree value is taken as noise from the tree model and the pixel is written as fully bare:

    tree = 0, non-vegetated = 100, non-tree vegetation = 0

Why: on deserts, ice and bare rock the tree model often leaves a few percent of tree cover.  Without this rule the sum cap
(tree + non-vegetated <= 100, applied earlier, lowers the non-vegetated value, never the tree value) would pull the confident
bare value down to make room for that noise.  Tree cover of 10 % or more is never changed, so sparse woodland is kept.

Applied to every tile, on land pixels only (all three layers valid, 0-100); water (200) and outside-projection (250) codes are
left as they are.  The raw bare value is the model's prediction before the sum cap.  From v1.4.0's bp10_apply.py (removed in
v1.5.2; at tag v1.5.1); values unchanged.  vcfPredict.py --noBarePriority switches it off, for testing only.
"""
import numpy as np

BARE_FULL = 100           # raw non-vegetated value at which the rule applies
TREE_MIN, TREE_MAX = 1, 9  # final tree cover changed by the rule (tree 0 needs no change)


def apply(tree: np.ndarray, bare: np.ndarray, nontree: np.ndarray, rawBare: np.ndarray) -> tuple:
    """The bare-priority rule on the final layers.

    tree, bare, nontree: the final tree, non-vegetated and non-tree vegetation layers (after the sum cap and the other rules);
    rawBare: the non-vegetated model's raw value, before the sum cap.  Returns (tree, bare, nontree, number of pixels changed);
    the inputs are not modified.
    """
    ok = lambda v: (v >= 0) & (v <= 100)
    land = ok(tree) & ok(bare) & ok(nontree)
    hit = land & (rawBare == BARE_FULL) & (tree >= TREE_MIN) & (tree <= TREE_MAX)
    tree, bare, nontree = tree.copy(), bare.copy(), nontree.copy()
    tree[hit] = 0
    bare[hit] = 100
    nontree[hit] = 0
    return tree, bare, nontree, int(hit.sum())
