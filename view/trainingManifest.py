import argparse
import logging
from pathlib import Path
import signal
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
# trainingManifest.py sample --labels <pattern with {tile}> --tiles <file> --trainingType pcttree|pctbare -y <year>
#                     -o <manifest.parq> --version <name> [--extra <parquet>] [--index range|position] [--notes text]
#   fresh-sample mode: the sample drawn from the label rasters (TrainingManifest.sample).
# trainingManifest.py build --manifest <manifest.parq> --predictors <file> --root <metrics root> [--roots <file>]
#                     -y <year> -o <split dir> [--tiles <file>] [--keysOnly] [--lowSun le0|s70|off] [--lowSunCheck on|off] [--workers N]
#   --roots: lines "<tile> <metrics root>" overriding --root for those tiles.
# trainingManifest.py accept --partner <partner split dir> --standard <standard split dir> --counts <V4r counts dir>
#                     --root <standard metrics root> [--roots <file>] -y <year> -o <report.json>
#   acceptance test of the blend A partner's split (TrainingManifest.accept); exit 1 on FAIL.
# -----------------------------------------------------------------------------
def main():

    # SIGTERM (a stopped run) ends the build through its clean-up path, which also ends the split's workers.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    ex = sub.add_parser('extract')
    ex.add_argument('--split', type=Path, required=True)
    ex.add_argument('-o', type=Path, required=True)
    ex.add_argument('--version', required=True)
    ex.add_argument('--notes', default='')

    sa = sub.add_parser('sample')
    sa.add_argument('--labels', required=True, help='Label file path with {tile}, e.g. <dir>/{tile}.samp.bin')
    sa.add_argument('--tiles', type=Path, required=True)
    sa.add_argument('--trainingType', choices=['pcttree', 'pctbare'], required=True)
    sa.add_argument('-y', type=int, required=True)
    sa.add_argument('-o', type=Path, required=True)
    sa.add_argument('--version', required=True)
    sa.add_argument('--extra', type=Path, default=None, help='Extra label rows (tid-year, x, y, label), appended.')
    sa.add_argument('--index', choices=['range', 'position'], default='range')
    sa.add_argument('--notes', default='')

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

    ac = sub.add_parser('accept')
    ac.add_argument('--partner', type=Path, required=True, help='Partner split (built from the V4r root, low-sun off).')
    ac.add_argument('--standard', type=Path, required=True, help='Standard split of the same manifest.')
    ac.add_argument('--counts', type=Path, required=True, help='Folder with <tile>_v4r_used_count.npy.')
    ac.add_argument('--root', type=Path, required=True, help='Standard metrics root (lowsun_counts.csv).')
    ac.add_argument('--roots', type=Path, default=None)
    ac.add_argument('-y', type=int, required=True)
    ac.add_argument('-o', type=Path, required=True)

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

    if args.cmd == 'sample':

        tiles = args.tiles.read_text().split()
        meta = TrainingManifest.sample({t: Path(args.labels.format(tile=t)) for t in tiles}, args.trainingType, args.y,
                                       args.o, args.version, args.extra, args.index, notes=args.notes)
        logger.info('Manifest ' + str(args.o) + ': ' +
                    ', '.join(p + ' ' + str(meta['parts'][p]['rows']) for p in meta['parts']))
        return 0

    roots = {}

    if args.roots:
        for line in args.roots.read_text().splitlines():
            if line.strip():
                t, r = line.split()
                roots[t] = Path(r)

    if args.cmd == 'accept':

        rep = TrainingManifest.accept(args.partner, args.standard, args.counts, lambda t: roots.get(t, args.root),
                                      args.y, args.o)
        logger.info('Partner split acceptance ' + rep['RESULT'] + ': ' + str(args.o))
        return 0 if rep['RESULT'] == 'PASS' else 1

    LowSun.configure(args.lowSun, None if args.lowSunCheck is None else args.lowSunCheck == 'on')
    tiles =args.tiles.read_text().split() if args.tiles else None

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
