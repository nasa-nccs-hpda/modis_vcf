#!/usr/bin/env python3
"""Build or check the pooled bare split (model/PooledSplit.py), the training table of the pooled bare model (2000-2023).

  python -m modis_vcf.view.buildPooledSplit build --years <years.tsv> --predictors docs/predictors_bare.txt -o <out> [--seed 0]
  python -m modis_vcf.view.buildPooledSplit check --years <years.tsv> -o <out>

<years.tsv>: one line per year, '<year><TAB><bare split dir>', the split dirs written by each year's run (split:bare,
<runRoot>/Python/splits/bare).  build writes <out> (refuses an existing folder) and then runs the check; exit 1 when the check
fails.  Then fit as for any model: view/fitAltModel.py --algo rf on <out>/xTrain.parq and <out>/yTrain.parq (docs/RUNNING.md).
"""
import argparse
import logging
from pathlib import Path
import sys

from modis_vcf.model import PooledSplit


def main():

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    for name in ('build', 'check'):
        p = sub.add_parser(name)
        p.add_argument('--years', type=Path, required=True, help='Lines <year><TAB><bare split dir>.')
        p.add_argument('-o', type=Path, required=True, help='The pooled split folder.')
        if name == 'build':
            p.add_argument('--predictors', type=Path, required=True, help='One predictor name per line.')
            p.add_argument('--seed', type=int, default=0, help='Year assignment seed (0 for the delivered model).')

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s %(message)s')
    logger = logging.getLogger('buildPooledSplit')
    splits = PooledSplit.readYears(args.years)

    if args.cmd == 'build':
        predictors = args.predictors.read_text().split()
        PooledSplit.build(splits, predictors, args.o, args.seed, logger)

    try:
        PooledSplit.check(args.o, splits, logger=logger)
    except ValueError as e:
        logger.error(str(e))
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
