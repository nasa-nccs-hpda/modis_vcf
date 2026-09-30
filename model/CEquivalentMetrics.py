"""
C-equivalent group-statistic metrics.

Reproduces, from the 72 "unsorted monthly" values of a pixel, the 324-value
metric vector that C's make_metrics.c (C_code/pge61/MOD_PR44B3/make_metrics.c)
feeds to the Weka models, so Python can use the same predictors as C.

C layout of the 324 values (band k: 0-6 = Band_1..Band_7, 7 = NDVI, 8 = Band_31;
period n: 0-7 = the eight monthly composites 2019065, 113, 161, 209, 257, 305,
353, 2020033):

    [  0,  72)  unsorted:                idx = k*8 + n
    [ 72, 126)  6 group stats x 9 bands: idx = 72  + f*9 + k    (unsorted periods)
    [126, 198)  per-band sorted, descending: idx = 126 + k*8 + s  (s = 0 max)
    [198, 261)  periods re-ordered by NDVI (descending), then
                max (198 + k) and 6 group stats: idx = 207 + f*9 + k
    [261, 324)  periods re-ordered by Band_31 (descending), same shape,
                max (261 + k), stats: idx = 270 + f*9 + k

Group stats f = 0..5 (C comments: mean13n, mean16n, mean19n, mean46n, mean79n,
amp15n), each on a (k, 8) array s:
    0: (s0 + s1) / 2            1: (s0 + s1 + s2 + s3) / 4
    2: (s0 + ... + s5) / 6      3: (s2 + s3) / 2
    4: (s4 + s5) / 2            5: (s0 + s1 - s2 - s3) / 2
with C integer semantics: shorts promoted to int, division truncating toward
zero, result stored in a short.

The NDVI/thermal re-ordering copies C's exchange sort exactly (comparison is
strict, so ties keep C's order), and moves all nine bands together.

All functions work on arrays shaped (9, 8, N): 9 bands in C order, 8 periods,
N pixels (any flattening).
"""

import numpy as np

C_BANDS = ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6',
           'Band_7', 'NDVI', 'Band_31']
NUM_BANDS = 9
NUM_PERIODS = 8
TOTAL_ATTRIB = 324

GROUP_STATS = ['Mean13', 'Mean16', 'Mean19', 'Mean46', 'Mean79', 'Amp15']

# Names of the blocks returned by computeBlocks, in C order of appearance
# (block name -> (C column of band 0, stride between bands)).
BLOCK_LAYOUT = {}
for _f, _s in enumerate(GROUP_STATS):
    BLOCK_LAYOUT['CUnsorted' + _s] = (72 + _f * 9, 1)
for _r in range(NUM_PERIODS):
    BLOCK_LAYOUT['CSortedRank' + str(_r)] = (126 + _r, 8)
BLOCK_LAYOUT['CGreenestMax'] = (198, 1)
for _f, _s in enumerate(GROUP_STATS):
    BLOCK_LAYOUT['CGreenest' + _s] = (207 + _f * 9, 1)
BLOCK_LAYOUT['CThermalMax'] = (261, 1)
for _f, _s in enumerate(GROUP_STATS):
    BLOCK_LAYOUT['CThermal' + _s] = (270 + _f * 9, 1)

BLOCK_NAMES = list(BLOCK_LAYOUT)


def cColumn(blockName: str, bandIndex: int) -> int:
    """C metric index (0-323) of a block's value for band index k."""
    first, stride = BLOCK_LAYOUT[blockName]
    return first + bandIndex * stride


def _cdiv(x: np.ndarray, d: int) -> np.ndarray:
    """C integer division of int32 values: truncate toward zero."""
    return np.sign(x) * (np.abs(x) // d)


def groupStats(s: np.ndarray) -> np.ndarray:
    """(9, 8, N) int16 -> (6, 9, N) int16, the six group statistics."""
    x = s.astype(np.int32)
    out = np.empty((6, NUM_BANDS, s.shape[2]), dtype=np.int16)
    out[0] = _cdiv(x[:, 0] + x[:, 1], 2)
    out[1] = _cdiv(x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3], 4)
    out[2] = _cdiv(x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3] + x[:, 4] + x[:, 5], 6)
    out[3] = _cdiv(x[:, 2] + x[:, 3], 2)
    out[4] = _cdiv(x[:, 4] + x[:, 5], 2)
    out[5] = _cdiv(x[:, 0] + x[:, 1] - x[:, 2] - x[:, 3], 2)
    return out


def exchangeSort(s: np.ndarray, keyBand: int) -> np.ndarray:
    """
    C's sort of the periods by one band's values, descending, moving all
    bands together:
        for n in 0..6: for nn in n+1..7:
            if s[key][n] < s[key][nn]: swap column n and nn for every band
    """
    s = s.copy()

    for n in range(NUM_PERIODS - 1):

        for nn in range(n + 1, NUM_PERIODS):

            swap = s[keyBand, n] < s[keyBand, nn]

            if swap.any():

                colN = s[:, n].copy()
                s[:, n] = np.where(swap, s[:, nn], colN)
                s[:, nn] = np.where(swap, colN, s[:, nn])

    return s


def computeBlocks(unsorted: np.ndarray) -> dict:
    """
    unsorted: (9, 8, N) int16, C band order and period order.
    Returns {blockName: (9, N) int16} for every name in BLOCK_NAMES.
    """
    assert unsorted.shape[:2] == (NUM_BANDS, NUM_PERIODS)
    unsorted = unsorted.astype(np.int16, copy=False)
    blocks = {}

    stats = groupStats(unsorted)

    for f, name in enumerate(GROUP_STATS):
        blocks['CUnsorted' + name] = stats[f]

    # Per-band sort, descending (ties are irrelevant: equal values).
    perBand = -np.sort(-unsorted.astype(np.int32), axis=1).astype(np.int16)

    for r in range(NUM_PERIODS):
        blocks['CSortedRank' + str(r)] = perBand[:, r]

    for prefix, key in (('CGreenest', 7), ('CThermal', 8)):

        ordered = exchangeSort(unsorted, key)
        blocks[prefix + 'Max'] = ordered[:, 0]
        stats = groupStats(ordered)

        for f, name in enumerate(GROUP_STATS):
            blocks[prefix + name] = stats[f]

    return blocks


def toCVector(blocks: dict) -> np.ndarray:
    """Assemble blocks into (324, N) int16 for the derived part (columns 72+)."""
    n = next(iter(blocks.values())).shape[1]
    vec = np.zeros((TOTAL_ATTRIB, n), dtype=np.int16)

    for name in BLOCK_NAMES:

        for k in range(NUM_BANDS):
            vec[cColumn(name, k)] = blocks[name][k]

    return vec
