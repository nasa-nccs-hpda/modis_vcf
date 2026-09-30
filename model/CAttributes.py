
# ----------------------------------------------------------------------------
# CAttributes
#
# The predictors that C's MOD_PR44B3 hands to its Weka models, and the names
# of the Python metrics that mirror them.
#
# C computes 324 metrics per pixel (MOD_PR44B3/make_metrics.c) and picks the
# ones its models use by index (MOD_PR44B3/MOD44B3.c, attributes[] and
# attributes_bare[]).  The two tables below are copied from those arrays and
# checked against the C source by the validation script, so they are not
# maintained by hand.
#
# Layout of the 324 (make_metrics.c), with k the band index 0..8 in the order
# Band_1 .. Band_7, NDVI, Band_31 and n a period 0..7:
#
#   [  0,  72)  unsorted period values           idx = k*8 + n
#   [ 72, 126)  six statistics of the unsorted   idx = 72 + stat*9 + k
#   [126, 198)  each band sorted, high to low    idx = 126 + k*8 + rank
#   [198, 261)  periods ordered by NDVI, high to low: the top period, then
#               six statistics                   idx = 198 + stat*9 + k
#   [261, 324)  periods ordered by Band_31, high to low: same shape
#                                                idx = 261 + stat*9 + k
# where the statistics are, in order, Mean13, Mean16, Mean19, Mean46, Mean79
# and Amp15 (see CMirrorMetrics).
# ----------------------------------------------------------------------------

# From MOD44B3.c, jint attributes[N_ATTRIB], N_ATTRIB = 64.
TREE_ATTRIBUTES = [
    0,   1,   2,   3,   4,   5,   6,   8,   9,   10,  12,  14,  15,  42,  44,  47,
    51,  52,  56,  59,  60,  61,  62,  63,  77,  79,  81,  87,  88,  90,  91,  96,
    100, 106, 108, 109, 123, 124, 190, 191, 199, 206, 207, 214, 215, 216, 223, 224,
    226, 233, 239, 242, 259, 270, 271, 275, 276, 277, 278, 286, 287, 294, 295, 296]

# From MOD44B3.c, jint attributes_bare[N_ATTRIB_BARE], N_ATTRIB_BARE = 21.
BARE_ATTRIBUTES = [
    14, 18, 53, 56, 60, 61, 62, 63, 64, 65, 66,
    67, 69, 70, 71, 86, 109, 111, 140, 189, 223]

C_BANDS = ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6',
           'Band_7', 'NDVI', 'Band_31']

STATS = ['Mean13', 'Mean16', 'Mean19', 'Mean46', 'Mean79', 'Amp15']

UNSORTED = 'UnsortedMonthlyBands'


# ----------------------------------------------------------------------------
# familyColumn
#
# The C metric index of a mirror family's band.
# ----------------------------------------------------------------------------
def familyColumn(family: str, band: str) -> int:

    k = C_BANDS.index(band)

    if family.startswith('CUns'):
        return 72 + STATS.index(family[4:]) * 9 + k

    if family.startswith('CSortedRank'):
        return 126 + k * 8 + int(family[len('CSortedRank'):])

    for prefix, start in (('CGreen', 198), ('CTherm', 261)):

        if family.startswith(prefix):

            tail = family[len(prefix):]
            stat = 0 if tail == 'Max' else 1 + STATS.index(tail)
            return start + stat * 9 + k

    raise ValueError('Unknown family: ' + family)


# ----------------------------------------------------------------------------
# allFamilies
# ----------------------------------------------------------------------------
def allFamilies() -> list:

    families = ['CUns' + s for s in STATS]
    families += ['CSortedRank' + str(r) for r in range(8)]
    families += ['CGreenMax'] + ['CGreen' + s for s in STATS]
    families += ['CThermMax'] + ['CTherm' + s for s in STATS]
    return families


# ----------------------------------------------------------------------------
# decode
#
# Turn a C metric index into ('unsorted', band, period) or (family, band).
# ----------------------------------------------------------------------------
def decode(index: int) -> tuple:

    if index < 72:
        return (UNSORTED, C_BANDS[index // 8], index % 8)

    for family in allFamilies():
        for band in C_BANDS:
            if familyColumn(family, band) == index:
                return (family, band)

    raise ValueError('Not a C metric index: ' + str(index))


# ----------------------------------------------------------------------------
# predictorName
#
# The name Python's Metrics.getMetricFromRf() understands for a C attribute.
# days is the list of the eight period day strings, e.g. '2019065'.
# ----------------------------------------------------------------------------
def predictorName(index: int, days: list) -> str:

    what = decode(index)

    if what[0] == UNSORTED:
        return UNSORTED + '-' + what[1] + '-Day_' + days[what[2]]

    return what[0] + '-' + what[1]


# ----------------------------------------------------------------------------
# predictorNames
# ----------------------------------------------------------------------------
def predictorNames(attributes: list, days: list) -> list:

    return [predictorName(i, days) for i in attributes]


# ----------------------------------------------------------------------------
# periodDays
#
# The eight period days for the given year: the first day of each group of
# three 16-day composites, the same days Metrics._daysSought uses.
# ----------------------------------------------------------------------------
def periodDays(year: int) -> list:

    days = [(year, 65), (year, 113), (year, 161), (year, 209), (year, 257),
            (year, 305), (year, 353), (year + 1, 33)]

    return [str(y) + str(d).zfill(3) for y, d in days]


# ----------------------------------------------------------------------------
# wantedFamilies
#
# {family: [bands]} for every attribute that is not an unsorted value, i.e.
# what the mirror metrics must produce for the tree cover and bare models.
# ----------------------------------------------------------------------------
def wantedFamilies() -> dict:

    wanted = {}

    for index in sorted(set(TREE_ATTRIBUTES + BARE_ATTRIBUTES)):

        what = decode(index)

        if what[0] != UNSORTED:
            wanted.setdefault(what[0], []).append(what[1])

    return wanted
