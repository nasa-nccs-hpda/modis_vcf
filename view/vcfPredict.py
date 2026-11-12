#!/usr/bin/python
"""Predict the 7 map layers of tiles for one processing year (model/VcfPredict.py) from their metrics and models.

Usage:

  python -m modis_vcf.view.vcfPredict --tc <models>/pcttree.bin --nv <models>/pctbare.bin -o <output dir>
      --metricsDir <metrics dir> -t h09v05 h11v02 -y 2019
"""


import argparse
from pathlib import Path
import sys

from modis_vcf.model.VcfPredict import MOD44_DIR
from modis_vcf.model import LowSun
from modis_vcf.model import RuleTile
from modis_vcf.model.VcfPredict import VcfPredict


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    """Command line: see the module docstring and --help."""
    desc = 'Use this application to run a VCF prediction.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-o',
                        type=Path,
                        required='True',
                        help='Write the predicted image here.')

    parser.add_argument('--metricsDir',
                        type=Path,
                        help='Path to existing metrics.')

    parser.add_argument('--modisDir',
                        type=Path,
                        default=MOD44_DIR,
                        help='Path to MODIS image directory.')

    parser.add_argument('--nv',
                        required=False,
                        type=Path,
                        help='Path to non-vegetated model (not needed for rule tiles, rows v15-v17).')

    parser.add_argument('--nvn',
                        required=False,
                        type=Path,
                        help='Path to non-vegetated top-N predictors.')

    parser.add_argument('-t',
                        type=str,
                        nargs='*',
                        help='A space-separated list of tile IDs in ' +
                             'the form h##v## h##v##')

    parser.add_argument('--tc',
                        required=False,
                        type=Path,
                        help='Path to tree-cover model (not needed for rule tiles, rows v15-v17).')

    parser.add_argument('--tcn',
                        required=False,
                        type=Path,
                        help='Path to tree-cover top-N predictors.')

    parser.add_argument('--nJobs',
                        type=int,
                        default=1,
                        help='Threads to use when predicting.')

    parser.add_argument('-y',
                        type=int,
                        required='True',
                        help='The year to run')

    parser.add_argument('--tcAgg',
                        default='mean',
                        choices=VcfPredict.AGG_METHODS,
                        help='How the tree-cover members are combined (default mean, which equals predict()).')

    parser.add_argument('--nvAgg',
                        default='mean',
                        choices=VcfPredict.AGG_METHODS,
                        help='How the non-vegetated members are combined (default mean).')

    parser.add_argument('--noSd',
                        action='store_true',
                        help='Do not write the Percent_Tree_Cover_SD and Percent_NonVegetated_SD layers.')


    parser.add_argument('--cloudFallbackTree',
                        type=Path,
                        default=None,
                        help='Cloud-fallback tree model (joblib file or compact .cf '
                             'directory).  With it, the tree layer of tiles in rows v06-v11 '
                             '(30N-30S) is blended with the cloud-fallback tree model (model/CloudFallbackBlend.py).')

    parser.add_argument('--cloudFallbackTreeTopN',
                        type=Path,
                        default=None,
                        help='The cloud-fallback tree model\'s top-N predictor file.')

    parser.add_argument('--cloudFallbackMetricsDir',
                        type=Path,
                        default=None,
                        help='Metrics root of the cloud-fallback composites (built with '
                             '--lowSun off; model/CloudFallback.py).')

    parser.add_argument('--counts',
                        type=Path,
                        default=None,
                        help='Directory with <tile>_counts_snow.npz (cloud counts CUR_ZS, '
                             'model/CloudFallback.py) for the year being mapped.')

    parser.add_argument('--noBarePriority',
                        action='store_true',
                        help='Testing only: do not apply the bare-priority rule (raw non-vegetated 100 and '
                             'tree 1-9 -> tree 0, non-vegetated 100; model/BarePriority.py), the last step of '
                             'the product since v1.4.0 (named BP10 there).')

    parser.add_argument('--lowSun',
                        choices=['le0', 's70', 'off'],
                        default=None,
                        help='Low-sun bright substitution: le0 (default: '
                             'solar zenith > 70 and <= 0), s70 (> 70 only) or '
                             'off (v1.3.0).')

    parser.add_argument('--lowSunCheck',
                        choices=['on', 'off'],
                        default=None,
                        help='Check cached dailies against the low-sun mode '
                             '(default on).  Probably unnecessary in '
                             'production, where every run builds fresh '
                             'metrics roots.')

    parser.add_argument('--format',
                        choices=VcfPredict.MAP_FORMATS,
                        default='h5',
                        help='Map output: h5 = one Collection 7 MOD44B file per tile and year (default), '
                             'geotiff = one GeoTIFF per layer, both.')

    parser.add_argument('--h5Attr',
                        action='append',
                        default=[],
                        metavar='NAME=VALUE',
                        help='Set a root attribute of the C7 file (e.g. PGEVersion=7.0.2); repeatable.')

    args = parser.parse_args()

    for a in args.h5Attr:
        if '=' not in a:
            parser.error('--h5Attr needs NAME=VALUE, not ' + a)

    h5Overrides = dict(a.split('=', 1) for a in args.h5Attr)

    # ---
    # Rule tiles (rows v15-v17, model/RuleTile.py) are written from MOD44W
    # alone; every other tile needs the four model files.
    # ---
    ruleTids = [t for t in args.t if RuleTile.isRuleTile(t)]
    tids = [t for t in args.t if not RuleTile.isRuleTile(t)]

    if tids and None in (args.tc, args.tcn, args.nv, args.nvn):
        parser.error('--tc, --tcn, --nv and --nvn are required for ' + ' '.join(tids))

    if ruleTids:
        VcfPredict.forRuleTiles(args.o, writeSd=not args.noSd, mapFormat=args.format,
                                h5Overrides=h5Overrides).run(tids=ruleTids, years=[args.y])

    if not tids:
        return 0

    # ---
    # Low-sun mode (passed to composite workers through the environment).
    # ---
    LowSun.configure(args.lowSun,
                     None if args.lowSunCheck is None
                     else args.lowSunCheck == 'on')

    vcfp = VcfPredict(args.tc,
                      args.tcn,
                      args.nv,
                      args.nvn,
                      args.o, 
                      args.metricsDir,
                      args.modisDir,
                      nJobs=args.nJobs,
                      treeCoverAgg=args.tcAgg,
                      nonvegAgg=args.nvAgg,
                      writeSd=not args.noSd,
                      cloudFallbackTreeFile=args.cloudFallbackTree,
                      cloudFallbackTreeTopNFile=args.cloudFallbackTreeTopN,
                      cloudFallbackMetricsDir=args.cloudFallbackMetricsDir,
                      countsDir=args.counts,
                      barePriority=not args.noBarePriority,
                      mapFormat=args.format,
                      h5Overrides=h5Overrides)

    vcfp.run(tids=tids, years=[args.y])


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
