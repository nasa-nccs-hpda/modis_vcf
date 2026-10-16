import csv
import json
import os
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# LowSun
#
# Low-sun bright substitution (PGE61 fix_bands.c's intent; LE0 in v1.4.0).
# Band_1-7 and Band_31 inputs with solar zenith above a threshold, and with
# LE0 also solz <= 0 (polar night), become BandDayFile.ZENITH_CLAMP (16000).
#
# Modes:
#   le0  threshold 70 + solz <= 0 (default since v1.4.1; v1.4.0's setting)
#   s70  threshold 70 only
#   off  no substitution (v1.3.0 behaviour; Band_31 keeps its own > 70 clamp)
#
# The mode travels in the environment (VCF_LOWSUN_T, VCF_LOWSUN_LE0), because
# composite workers are spawned processes.  configure() sets it; setting the
# variables by hand still works (deprecated, kept for one version).
# VCF_LOWSUN_TILES (comma list) limits the tiles, as before.
#
# Cache guard: dailies in 1-Days are cached and reused whatever the current
# mode, which is how 34 v1.4.0 tiles kept unsubstituted inputs.  guard()
# stamps each tile-year root with lowsun.json; a root without a stamp is
# checked from its cached Band_1 dailies against solar zenith, and a
# mismatch stops the run.  The check can be switched off (--lowSunCheck off,
# VCF_LOWSUN_CHECK=0); in production, where every run builds fresh roots,
# it is probably unnecessary.  guard() also writes lowsun_counts.csv (pixels
# substituted per input day) once per root.
# ----------------------------------------------------------------------------
MODES = {'le0': (70, True), 's70': (70, False), 'off': (0, False)}
DEFAULT_MODE = 'le0'
ENV_T = 'VCF_LOWSUN_T'
ENV_LE0 = 'VCF_LOWSUN_LE0'
ENV_TILES = 'VCF_LOWSUN_TILES'
ENV_CHECK = 'VCF_LOWSUN_CHECK'
STAMP = 'lowsun.json'
COUNTS = 'lowsun_counts.csv'
CLAMP = 16000

# Roots already guarded in this process (VcfPredict builds Metrics twice).
_guarded = {}


# ----------------------------------------------------------------------------
# configure
# ----------------------------------------------------------------------------
def configure(mode: str = None, check: bool = None) -> None:

    if mode is not None:

        if mode not in MODES:
            raise ValueError('Low-sun mode must be one of ' + str(list(MODES)))

        t, le0 = MODES[mode]
        os.environ[ENV_T] = str(t)
        os.environ[ENV_LE0] = '1' if le0 else '0'

    if check is not None:
        os.environ[ENV_CHECK] = '1' if check else '0'


# ----------------------------------------------------------------------------
# configureFromArgv
# ----------------------------------------------------------------------------
def configureFromArgv(argv: list) -> list:
    """For scripts that read sys.argv by position: take out --lowSun <mode>
    and --lowSunCheck on|off, configure, and return the remaining args."""

    rest, mode, check, i = [], None, None, 0

    while i < len(argv):

        if argv[i] in ('--lowSun', '--lowSunCheck') and i + 1 < len(argv):

            if argv[i] == '--lowSun':
                mode = argv[i + 1]
            else:
                check = argv[i + 1] == 'on'

            i += 2
            continue

        rest.append(argv[i])
        i += 1

    configure(mode, check)
    return rest


# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------
def settings() -> tuple:
    """(threshold, le0) in force.  Threshold 0 = no substitution."""

    t = os.environ.get(ENV_T, '')

    if t == '':

        t, le0 = MODES[DEFAULT_MODE]
        le = os.environ.get(ENV_LE0, '')
        return t, le0 if le == '' else le == '1'

    return int(t), os.environ.get(ENV_LE0, '0') == '1'


# ----------------------------------------------------------------------------
# modeName
# ----------------------------------------------------------------------------
def modeName(t: int = None, le0: bool = None) -> str:

    if t is None:
        t, le0 = settings()

    if t <= 0:
        return 'off'

    for name, (mt, mle0) in MODES.items():
        if (mt, mle0) == (t, le0):
            return name

    return 't' + str(t) + ('_le0' if le0 else '')


# ----------------------------------------------------------------------------
# appliesTo
# ----------------------------------------------------------------------------
def appliesTo(tid: str) -> bool:

    t, _ = settings()
    tiles = os.environ.get(ENV_TILES, '')
    return t > 0 and (not tiles or tid in tiles.split(','))


# ----------------------------------------------------------------------------
# mask
# ----------------------------------------------------------------------------
def mask(solz: np.ndarray, t: int = None, le0: bool = None) -> np.ndarray:

    if t is None:
        t, le0 = settings()

    if t <= 0:
        return np.zeros(solz.shape, dtype=bool)

    low = solz > t

    if le0:
        low |= solz <= 0

    return low


# ----------------------------------------------------------------------------
# guard
# ----------------------------------------------------------------------------
def guard(metrics) -> None:
    """Stamp, check and count the low-sun mode of one Metrics root."""

    root = metrics._outDir
    key = str(root)
    t, le0 = settings()
    tid = metrics._tid
    effective = modeName(t, le0) if appliesTo(tid) else 'off'

    if _guarded.get(key) == effective:
        return

    log = metrics._logger
    stampFile = root / STAMP
    log.info('Low-sun mode: ' + effective)

    if stampFile.exists():

        stamped = json.loads(stampFile.read_text()).get('mode')

        if stamped != effective:

            raise RuntimeError('Low-sun mismatch: ' + str(root) +
                               ' was built with mode ' + str(stamped) +
                               ', this run uses ' + effective + '.')

        _guarded[key] = effective
        return

    days = sorted(metrics._dayDir.glob('*-' + tid + '-*-' +
                                       metrics._productType.BAND1 + '.bin'))
    checked = 0

    if os.environ.get(ENV_CHECK, '1') != '0':

        for f in days:

            found = _detect(metrics, f)

            if found is None:
                continue

            checked += 1

            if effective not in found:

                raise RuntimeError('Low-sun mismatch: cached daily ' +
                                   f.name + ' fits mode(s) ' +
                                   (', '.join(sorted(found)) or 'none') +
                                   ', this run uses ' + effective +
                                   '.  Rebuild the root, or switch the '
                                   'check off (--lowSunCheck off).')

        if not days and any(metrics._compDir.glob('*.bin')):

            raise RuntimeError('Low-sun check: ' + str(root) + ' has '
                               'composites but no Band_1 dailies to check. '
                               'Rebuild the root, or switch the check off '
                               '(--lowSunCheck off).')

    try:

        _writeCounts(metrics, root / COUNTS, t, le0, effective)

        tmp = stampFile.with_suffix('.tmp' + str(os.getpid()))
        tmp.write_text(json.dumps({'mode': effective, 'threshold': t,
                                   'le0': le0, 'tile': tid,
                                   'dailiesChecked': checked,
                                   'dailiesCached': len(days)}, indent=1))
        tmp.replace(stampFile)

    except OSError as e:

        # Earlier runs' roots are bound read-only: check, but do not stamp.
        log.warning('Low-sun stamp not written (' + str(e) + ')')

    _guarded[key] = effective


# ----------------------------------------------------------------------------
# _detect
# ----------------------------------------------------------------------------
def _detect(metrics, dayFile: Path):
    """Set of modes a cached Band_1 daily is consistent with, or None when it
    has no low-sun pixels (every mode gives the same file)."""

    from modis_vcf.model.BandDayFile import BandDayFile

    pt = metrics._productType
    yd = dayFile.name.split('-')[-2]
    bdf = BandDayFile().initFromParams(pt, pt.BAND1, metrics._tid,
                                       int(yd[:4]), int(yd[4:]),
                                       metrics._dayDir, metrics._logger)
    solz = bdf.getSolz()
    cached = np.fromfile(dayFile, dtype=bdf._dtype).reshape(solz.shape)

    def state(m):
        # True: all substituted; False: (nearly) none; 'mixed': neither.
        if not m.any():
            return None

        share = np.mean(cached[m] == CLAMP)
        return True if share == 1.0 else False if share < 0.99 else 'mixed'

    hi, night = state(solz > 70), state(solz <= 0)

    if hi is None and night is None:
        return None

    fits = set()

    if hi in (None, False) and night in (None, False):
        fits.add('off')

    if hi in (None, True) and night in (None, False):
        fits.add('s70')

    if hi in (None, True) and night in (None, True):
        fits.add('le0')

    return fits


# ----------------------------------------------------------------------------
# _writeCounts
# ----------------------------------------------------------------------------
def _writeCounts(metrics, outFile: Path, t: int, le0: bool,
                 effective: str) -> None:

    if outFile.exists():
        return

    from modis_vcf.model.BandDayFile import BandDayFile

    pt = metrics._productType
    year = metrics._year
    days = [(year, d) for d in pt.yearOneDays] + \
           [(year + 1, d) for d in pt.yearTwoDays]

    rows = []

    for y, d in days:

        bdf = BandDayFile().initFromParams(pt, pt.BAND1, metrics._tid, y, d,
                                           metrics._dayDir, metrics._logger)
        try:
            solz = bdf.getSolz()

        except Exception as e:

            metrics._logger.warning('Low-sun counts: no solz for ' + str(y) +
                                    str(d).zfill(3) + ': ' + str(e))
            rows.append([metrics._tid, y, d, effective, -1, -1, -1])
            continue

        n = int(mask(solz, t, le0).sum()) if effective != 'off' else 0
        rows.append([metrics._tid, y, d, effective,
                     int((solz > 70).sum()), int((solz <= 0).sum()), n])

    tmp = outFile.with_suffix('.tmp' + str(os.getpid()))

    with open(tmp, 'w', newline='') as f:

        w = csv.writer(f, lineterminator='\n')
        w.writerow(['tile', 'year', 'day', 'mode', 'solz_gt70', 'solz_le0',
                    'substituted'])
        w.writerows(rows)

    tmp.replace(outFile)
