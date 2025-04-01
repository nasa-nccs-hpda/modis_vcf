#!/usr/bin/python

import argparse
from pathlib import Path
import sys

from modis_vcf.model.VcfPredict import MOD44_DIR
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

    parser.add_argument('-t',
                        type=str,
                        nargs='*',
                        help='A space-separated list of tile IDs in ' +
                             'the form h##v## h##v##')

    parser.add_argument('--tc',
                        required=True,
                        type=Path,
                        help='Path to tree-cover model.')

    parser.add_argument('-y',
                        type=int,
                        required='True',
                        help='The year to run')

    args = parser.parse_args()
    
    vcfp = VcfPredict(args.tc,
                      args.nv,
                      args.o, 
                      args.metricsDir,
                      args.modisDir)

    vcfp.run(tids=args.t, years=[args.y])


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
