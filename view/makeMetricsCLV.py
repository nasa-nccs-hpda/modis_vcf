#!/usr/bin/python

import argparse
import logging
from pathlib import Path
import sys

from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.Pair import Pair


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/makeMetricsCLV.py -o /explore/nobackup/people/rlgill/SystemTesting/modis-vcf 
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to compute metrics.'
    parser = argparse.ArgumentParser(description=desc)

    inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')

    parser.add_argument('-i',
                        type=Path,
                        default=inDir,
                        help='Input directory for finding MODIS files')

    parser.add_argument('-m',
                        type=str,
                        nargs='*',
                        help='A space-separated list of metric names')

    parser.add_argument('-o',
                        type=Path,
                        default='.',
                        help='Output directory for writing metrics')

    parser.add_argument('-p',
                        action='store_true',
                        help='Print descriptions of available metrics.\n' + \
                             'This causes other args. to be ignored.')

    parser.add_argument('-t',
                        type=str,
                        default='h09v05',
                        help='Tile ID in the form h##v##')

    parser.add_argument('-y',
                        type=int,
                        default=2019,
                        help='The year to run')

    args = parser.parse_args()
    
    # ---
    # Logging
    # ---
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # ---
    # Make Metrics
    # ---
    mm = Metrics(args.t, args.y, args.i, args.o, logger=logger)
    
    if args.p:

        print(mm.availableMetrics)
        sys.exit(1)
        
    metricsToRun = args.m or mm.availableMetrics
    
    for metricName in metricsToRun:
        mm.getMetric(metricName)
    

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
