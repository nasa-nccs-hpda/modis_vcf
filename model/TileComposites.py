"""The 8 composite periods of one tile and processing year from C7 dailies (v2.0.0), ready for the metrics step.

buildTile writes the composites the metrics step reads into the tile-year root of the metrics folder
(<metrics root>/<tile>/<year>/2-Composites, plus diag/), then composites.json, which records where the composites came from:

  {"input": "dailies", "product": "MOD44B" | "MCD44B", "sensors": ["terra"] | ["terra", "aqua"], "aquaFill": bool,
   "options": {CompositeOptions}, "periods": {"2019065": {"aqua": true | false}, ...}, "code": "modis_vcf <version>"}

The rest of the chain reads it (source()): Metrics leaves out the zenith-only NDVI rule and LowSun the cache guard (both
belong to MOD44C inputs), and VcfPredict names the product (MCD44B when Aqua filled) and writes the Aqua_Fill layer.  A
tile-year root without composites.json is a MOD44C root and behaves as in v1.9.0.

Terra only: each period is composited straight into the tile-year root.
Aqua fill: Terra and Aqua are composited into <work root>/terra|aqua/<tile>/<year>, then merged (AquaFill) into the tile-year
root.  A period without Aqua dailies for the tile is the Terra composite (no fill).  Whether a run uses Aqua at all is decided
per processing year (aquaAvailable; lead 2026-11-19: no Aqua dailies for the year -> MOD44B with a warning; Aqua dailies for
the year -> every tile MCD44B, also where Aqua fills nothing).

Each period is a separate process (about 20 minutes and 23 GB per period and sensor on 2019 tiles); completed periods
(diag summary.json) are not redone.
"""

import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from modis_vcf.model import AquaFill, DailiesComposite as DC
from modis_vcf.model import Sensor as S

SOURCE = 'composites.json'
CODE = 'modis_vcf 2.0.0'


def source(tileYearRoot) -> dict:
    """composites.json of a tile-year root, or None (a MOD44C root)."""
    p = Path(tileYearRoot) / SOURCE
    return json.loads(p.read_text()) if p.exists() else None


def isDailies(tileYearRoot) -> bool:
    """True when the tile-year root holds composites from dailies."""
    s = source(tileYearRoot)
    return bool(s) and s.get('input') == 'dailies'


def aquaAvailable(dailiesDir: str, year: int) -> bool:
    """True when the dailies folder has any Aqua GA file (any tile) dated within the processing year."""
    days = {d.strftime('%Y%j') for p in DC.periods(year) for d in DC.daysOf(p)}
    prefix = S.AQUA.ga + '.A'

    for y in (year, year + 1):
        d = Path(dailiesDir) / S.AQUA.ga / str(y)

        if not d.is_dir():
            continue

        with os.scandir(d) as it:
            for e in it:
                if e.name.startswith(prefix) and e.name[len(prefix):len(prefix) + 7] in days:
                    return True

    return False


def _one(args):
    """One sensor-period composite in a worker process."""
    tile, period, dailiesDir, out, sensorName, options, diag = args
    if not DC.isDone(out, tile, period):
        DC.composite(tile, period, dailiesDir, out, S.get(sensorName), DC.CompositeOptions(**options), diag)
    return sensorName, period


def buildTile(tile: str, year: int, dailiesDir: str, tileYearRoot, workRoot=None, aquaFill: bool = False,
              options: DC.CompositeOptions = DC.CompositeOptions(), workers: int = 4,
              logger: logging.Logger = None, diag: bool = False) -> dict:
    """Composite the tile's 8 periods (module docstring) and return composites.json's content.  diag: write the full
    per-pixel diagnostics (off by default, lead 2026-11-19; about 7 GB per tile with Aqua fill)."""
    log = logger or logging.getLogger(__name__)
    root = Path(tileYearRoot)
    want = dict(input='dailies', product='MCD44B' if aquaFill else 'MOD44B',
                sensors=['terra', 'aqua'] if aquaFill else ['terra'], aquaFill=aquaFill, options=asdict(options))
    have = source(root)

    if have is not None:
        if {k: have.get(k) for k in want} != want:
            raise RuntimeError('%s was built with %s, this run asks for %s: use a new metrics root.' %
                               (root / SOURCE, {k: have.get(k) for k in want}, want))
        log.info('Composites of %s %d already built (%s).' % (tile, year, have['product']))
        return have

    ps = DC.periods(year)
    root.mkdir(parents=True, exist_ok=True)

    if aquaFill:
        if workRoot is None:
            raise ValueError('Aqua fill needs a work root for the Terra and Aqua composites.')
        dirs = {s: str(Path(workRoot) / s / tile / str(year)) for s in ('terra', 'aqua')}
        aqua = {p: DC.hasDailies(dailiesDir, S.AQUA, tile, p) for p in ps}
    else:
        dirs = {'terra': str(root)}
        aqua = {p: False for p in ps}

    jobs = [(tile, p, dailiesDir, dirs['terra'], 'terra', asdict(options), diag) for p in ps] + \
           [(tile, p, dailiesDir, dirs['aqua'], 'aqua', asdict(options), diag) for p in ps if aqua[p]]
    jobs = [j for j in jobs if not DC.isDone(j[3], tile, j[1])]
    log.info('Compositing %s %d from dailies: %d sensor-periods to build, %d workers%s.' %
             (tile, year, len(jobs), workers, ', Aqua fill' if aquaFill else ''))

    if aquaFill and not all(aqua.values()):
        log.warning('%s %d: no Aqua dailies in period(s) %s; those periods are Terra only.' %
                    (tile, year, ', '.join(p for p in ps if not aqua[p])))

    if jobs:
        if workers <= 1:
            for j in jobs:
                _one(j)
        else:
            with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=multiprocessing.get_context('spawn')) as ex:
                for sensorName, p in ex.map(_one, jobs):
                    log.info('Composite %s %s %s done.' % (tile, p, sensorName))

    if aquaFill:
        for p in ps:
            if not DC.isDone(str(root), tile, p):
                s = AquaFill.merge(tile, p, dirs['terra'], dirs['aqua'] if aqua[p] else None, str(root), diag=diag)
                log.info('Aqua fill %s %s: %.4f of the tile.' % (tile, p, s['fill_share']))

    have = dict(want, diag=diag, periods={p: {'aqua': aqua[p]} for p in ps}, inputs=_inputs(tile, ps, dirs, aqua), code=CODE)
    tmp = root / (SOURCE + '.part')
    tmp.write_text(json.dumps(have, indent=1))
    os.replace(tmp, root / SOURCE)
    return have


def _inputs(tile: str, ps: list, dirs: dict, aqua: dict) -> dict:
    """Per daily product: number of files used, first and last file name (from the sensor-period summaries)."""
    names = []

    for p in ps:
        for s_ in ('terra', 'aqua'):
            if s_ == 'aqua' and not aqua[p]:
                continue
            names += json.loads(Path(DC.diagName(dirs[s_], tile, p, 'summary.json')).read_text()).get('inputs', [])

    out = {}

    for n in sorted(set(names)):
        e = out.setdefault(n.split('.')[0], dict(files=0, first=n, last=n))
        e['files'] += 1
        e['last'] = n

    return out


def inputSummary(tileYearRoot) -> list:
    """InputDataProducts entries for a root of composites from dailies, one per daily product: '<product> C7 daily x<n>
    (<first file> ... <last file>)'.  Empty for a MOD44C root."""
    s = source(tileYearRoot) or {}
    return ['%s daily x%d (%s ... %s)' % (k, v['files'], v['first'], v['last']) for k, v in sorted(s.get('inputs', {}).items())]


def aquaFillBits(tileYearRoot, tile: str, year: int):
    """The Aqua_Fill layer: uint8, bit n set where period n (period 1 in bit 0, Metrics order) was filled from Aqua (diag
    fill_src of the merge).  None for a root without Aqua fill."""
    import numpy as np

    s = source(tileYearRoot)

    if not s or not s.get('aquaFill'):
        return None

    bits = np.zeros((DC.N, DC.N), np.uint8)

    for n, p in enumerate(DC.periods(year)):
        bits |= (np.load(DC.diagName(str(tileYearRoot), tile, p, 'fill_src')).astype(np.uint8) & 1) << n

    return bits
