#!/usr/bin/python

import argparse
import logging
from pathlib import Path
import sys

from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.ProductTypeMod09A import ProductTypeMod09A
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/makeMetricsCLV.py -o /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles/MOD44C -t h09v05 -y 2019
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to compute metrics.'
    parser = argparse.ArgumentParser(description=desc)

    inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')
    inDir09 = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD09A1')

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
                        help='Print descriptions of available metrics ' + \
                             'and exit.')

    parser.add_argument('-t',
                        nargs='+',
                        type=str,
                        required='True',
                        help='List of tile IDs in the form h##v## h##v## ...')

    parser.add_argument('-y',
                        type=int,
                        required='True',
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
    # prodType = ProductTypeMod44(args.i)
    prodType = ProductTypeMod09A(inDir09, inDir)
    
    for tid in args.t:

        mm = Metrics(tid, args.y, prodType, args.o, logger=logger)
    
        if args.p:

            print(mm.availableMetrics)
            sys.exit(1)
        
        metricsToRun = args.m or mm.availableMetrics
    
        for metricName in metricsToRun:
        
            logger.info('Attempting ' + metricName)
            mm.getMetric(metricName)
    

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
