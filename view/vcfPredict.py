#!/usr/bin/python

import argparse
from pathlib import Path
import sys

from modis_vcf.model.VcfPredict import MOD44_DIR
from modis_vcf.model import LowSun
from modis_vcf.model.VcfPredict import VcfPredict


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/vcfPredict.py --tc /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/SystemTests/3-Models/pcttree.bin --nv /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/SystemTests/3-Models/pctbare.bin -o /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/SystemTests/4-VcfProcess --metricsDir /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/SystemTests/1-Metrics -t h09v05 h11v02 -y 2019
# -----------------------------------------------------------------------------
def main():
    
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
                        required=True,
                        type=Path,
                        help='Path to non-vegetated model.')

    parser.add_argument('--nvn',
                        required=True,
                        type=Path,
                        help='Path to non-vegetated top-N predictors.')

    parser.add_argument('-t',
                        type=str,
                        nargs='*',
                        help='A space-separated list of tile IDs in ' +
                             'the form h##v## h##v##')

    parser.add_argument('--tc',
                        required=True,
                        type=Path,
                        help='Path to tree-cover model.')

    parser.add_argument('--tcn',
                        required=True,
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


    parser.add_argument('--lowSun',
                        choices=['le0', 's70', 'off'],
                        default=None,
                        help='Low-sun bright substitution: le0 (default: '
                             'solar zenith > 70 and <= 0), s70 (> 70 only) or '
                             'off (v1.3.0).  Unset keeps VCF_LOWSUN_* '
                             'variables if set (deprecated).')

    parser.add_argument('--lowSunCheck',
                        choices=['on', 'off'],
                        default=None,
                        help='Check cached dailies against the low-sun mode '
                             '(default on).  Probably unnecessary in '
                             'production, where every run builds fresh '
                             'metrics roots.')

    args = parser.parse_args()
    
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
                      writeSd=not args.noSd)

    vcfp.run(tids=args.t, years=[args.y])


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
