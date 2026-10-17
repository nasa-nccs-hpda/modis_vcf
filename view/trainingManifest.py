import argparse
import logging
from pathlib import Path
import sys

from modis_vcf.model import LowSun
from modis_vcf.model import TrainingManifest
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
from modis_vcf.model.VcfPredict import MOD44_DIR


# -----------------------------------------------------------------------------
# main
#
# Training sample as a key manifest (model/TrainingManifest.py).
#
# trainingManifest.py extract --split <split dir> -o <manifest.parq> --version v1.4.0 [--notes text]
# trainingManifest.py build --manifest <manifest.parq> --predictors <file> --root <metrics root> [--roots <file>]
#                     -y <year> -o <split dir> [--tiles <file>] [--keysOnly] [--lowSun le0|s70|off] [--lowSunCheck on|off] [--workers N]
#   --roots: lines "<tile> <metrics root>" overriding --root for those tiles.
# -----------------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    ex = sub.add_parser('extract')
    ex.add_argument('--split', type=Path, required=True)
    ex.add_argument('-o', type=Path, required=True)
    ex.add_argument('--version', required=True)
    ex.add_argument('--notes', default='')

    bu = sub.add_parser('build')
    bu.add_argument('--manifest', type=Path, required=True)
    bu.add_argument('--predictors', type=Path, required=True)
    bu.add_argument('--root', type=Path, required=True)
    bu.add_argument('--roots', type=Path, default=None)
    bu.add_argument('-y', type=int, required=True)
    bu.add_argument('-o', type=Path, required=True)
    bu.add_argument('--tiles', type=Path, default=None)
    bu.add_argument('--keysOnly', action='store_true')
    bu.add_argument('--lowSun', choices=['le0', 's70', 'off'], default=None)
    bu.add_argument('--lowSunCheck', choices=['on', 'off'], default=None)
    bu.add_argument('--workers', type=int, default=1, help='Processes computing the tiles (same split for any number).')

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if args.cmd == 'extract':

        meta = TrainingManifest.extract(args.split, args.o, args.version, args.notes)
        logger.info('Manifest ' + str(args.o) + ': ' +
                    ', '.join(p + ' ' + str(meta['parts'][p]['rows']) for p in meta['parts']))
        return 0

    LowSun.configure(args.lowSun, None if args.lowSunCheck is None else args.lowSunCheck == 'on')
    roots = {}

    if args.roots:
        for line in args.roots.read_text().splitlines():
            if line.strip():
                t, r = line.split()
                roots[t] = Path(r)

    tiles = args.tiles.read_text().split() if args.tiles else None

    TrainingManifest.build(args.manifest,
                           args.predictors.read_text().split(),
                           lambda t: roots.get(t, args.root),
                           args.y,
                           ProductTypeMod44(MOD44_DIR),
                           args.o,
                           logger,
                           tiles=tiles,
                           keysOnly=args.keysOnly,
                           workers=args.workers)
    return 0


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
