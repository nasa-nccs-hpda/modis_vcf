#!/usr/bin/python
"""Build composites from C7 dailies (v2.0.0, B2 rules) without the year driver.

Usage:

  # the 8 periods of tiles, ready for the metrics step (model/TileComposites.py)
  python -m modis_vcf.view.makeComposites tile -o <metrics dir> -y 2019 -t h09v05 h12v09 --dailiesDir <dir> \
      [--aquaFill --work <dir>] [--workers 4] [--option polarFill=false ...]

  # one sensor and period (model/DailiesComposite.py): <out>/2-Composites and <out>/diag
  python -m modis_vcf.view.makeComposites period h09v05 2019161 --dailiesDir <dir> --out <dir> [--sensor aqua] [--option ...]

  # Aqua fill of one period from finished Terra and Aqua period outputs (model/AquaFill.py)
  python -m modis_vcf.view.makeComposites merge h09v05 2019161 --terra <dir> --aqua <dir> --out <dir> [--b6rule b]

Options (--option name=value, repeatable): polarFill, otherOrbit, f2Rank, f3Guard (true / false), f3 (mean3 / single),
rules (b1 / b0).  The defaults are B2.
"""

import argparse
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path

from modis_vcf.model import AquaFill, DailiesComposite as DC, Sensor, TileComposites


def _options(pairs: list) -> DC.CompositeOptions:
    kinds = {f.name: f.type for f in fields(DC.CompositeOptions)}
    kw = {}

    for p in pairs:
        if '=' not in p:
            raise SystemExit('--option needs name=value, not ' + p)
        k, v = p.split('=', 1)
        if k not in kinds:
            raise SystemExit('Unknown option ' + k + '; known: ' + ', '.join(kinds))
        if kinds[k] in (bool, 'bool'):
            if v.lower() not in ('true', 'false'):
                raise SystemExit(k + ' must be true or false')
            kw[k] = v.lower() == 'true'
        else:
            kw[k] = v

    return DC.CompositeOptions(**kw)


def main():
    """Command line: see the module docstring and --help."""
    parser = argparse.ArgumentParser(description='Composites from C7 dailies (B2 rules).')
    sub = parser.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('tile', help='the 8 periods of tiles, into a metrics root')
    t.add_argument('-o', type=Path, required=True, help='metrics root (<o>/<tile>/<year>/2-Composites)')
    t.add_argument('-y', type=int, required=True, help='processing year')
    t.add_argument('-t', nargs='+', required=True, help='tiles')
    t.add_argument('--dailiesDir', required=True)
    t.add_argument('--aquaFill', action='store_true', help='Aqua good days fill Terra pixels with no good day (MCD44B)')
    t.add_argument('--work', type=Path, help='work root for the Terra and Aqua composites (needed with --aquaFill)')
    t.add_argument('--workers', type=int, default=4, help='sensor-periods built at once (about 23 GB each)')
    t.add_argument('--option', action='append', default=[])
    t.add_argument('--diag', action='store_true', help='also write the per-pixel diagnostics (off by default)')

    p = sub.add_parser('period', help='one sensor and period')
    p.add_argument('tile')
    p.add_argument('period', help='yyyyddd')
    p.add_argument('--dailiesDir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--sensor', default='terra', choices=sorted(Sensor.SENSORS))
    p.add_argument('--option', action='append', default=[])
    p.add_argument('--diag', action='store_true', help='also write the per-pixel diagnostics (off by default)')

    m = sub.add_parser('merge', help='Aqua fill of one period')
    m.add_argument('tile')
    m.add_argument('period')
    m.add_argument('--terra', required=True)
    m.add_argument('--aqua', help='Aqua period output; omit when the period has no Aqua dailies')
    m.add_argument('--out', required=True)
    m.add_argument('--b6rule', default='a', choices=['a', 'b'])
    m.add_argument('--diag', action='store_true', help='also carry solz_used / vz_used / gooddays / group')

    a = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s %(levelname)s %(message)s')

    if a.cmd == 'tile':
        if a.aquaFill and not TileComposites.aquaAvailable(a.dailiesDir, a.y):
            logging.warning('--aquaFill: no Aqua dailies for processing year %d; Terra only (MOD44B).' % a.y)
            a.aquaFill = False
        for tile in a.t:
            s = TileComposites.buildTile(tile, a.y, a.dailiesDir, a.o / tile / str(a.y), a.work, a.aquaFill,
                                         _options(a.option), a.workers, diag=a.diag)
            print(tile, s['product'], flush=True)
    elif a.cmd == 'period':
        s = DC.composite(a.tile, a.period, a.dailiesDir, a.out, Sensor.get(a.sensor), _options(a.option), a.diag)
        print(json.dumps({k: v for k, v in s.items() if k != 'inputs'}), flush=True)
    else:
        s = AquaFill.merge(a.tile, a.period, a.terra, a.aqua, a.out, a.b6rule, a.diag)
        print(json.dumps(s), flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
