#!/usr/bin/python

import argparse
from pathlib import Path
import sys

from modis_vcf.model.VcfPredict import MOD44_DIR
from modis_vcf.model.VcfPredict import VcfPredict


# -----------------------------------------------------------------------------
# main
#
# python modis_vcf/view/vcfPredict.py -o /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/MOD44/vcfProcess --modelFile /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/MOD44-save/Test-MC-model.bin --metricsDir /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/MOD44 -t h09v05 -y 2019
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

    parser.add_argument('--modelFile',
                        required=True,
                        type=Path,
                        help='Path to trained model.')

    parser.add_argument('--modisDir',
                        type=Path,
                        default=MOD44_DIR,
                        help='Path to MODIS image directory.')

    parser.add_argument('-t',
                        type=str,
                        nargs='*',
                        help='A space-separated list of tile IDs in ' +
                             'the form h##v## h##v##')

    parser.add_argument('-y',
                        type=int,
                        required='True',
                        help='The year to run')

    args = parser.parse_args()
    
    vcfp = VcfPredict(args.modelFile, 
                      args.o, 
                      args.metricsDir,
                      args.modisDir)

    vcfp.run(tids=args.t, years=[args.y])


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
