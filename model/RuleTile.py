# ----------------------------------------------------------------------------
# RuleTile
#
# Tiles mapped by rule instead of by the models: rows v15-v17 (Antarctica,
# south of about 60 S).  There is no MOD44C for them, so no metrics and no
# prediction.  MOD44W marks land, water and out-of-projection; land is tree
# cover 0, non-vegetated 100 and non-tree vegetation 0.  The SD layers are 0
# (no members).  Quality and Cloud have every period's bit set on land, as on
# a v00 tile's periods without input (a period with no input day falls back
# and has no clear day); 0 on water and out-of-projection, as on every tile.
# VcfPredict writes the 7 layers (VcfPredict.forRuleTiles); YearRun gives a
# rule tile a map task with no metrics and no model dependencies.
# ----------------------------------------------------------------------------
VTILES = (15, 16, 17)

# The 8 composite periods of a year, one bit each (Metrics.getPeriodFlags).
ALL_PERIODS = 0xFF


# ----------------------------------------------------------------------------
# isRuleTile
# ----------------------------------------------------------------------------
def isRuleTile(tid: str) -> bool:
    """True for a tile ID such as 'h16v16' in rows v15-v17."""
    return int(tid[tid.index('v') + 1:]) in VTILES
