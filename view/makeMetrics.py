#!/usr/bin/python

import argparse
import logging
from pathlib import Path
import sys

from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.Metrics import Metrics
from modis_vcf.model import LowSun
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
from modis_vcf.model.Settings import MOD44_DIR


# -----------------------------------------------------------------------------
# main
#
# python -m modis_vcf.view.makeMetrics -o <metrics dir> -y 2019 -t h09v05 h11v02
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to compute metrics.'
    parser = argparse.ArgumentParser(description=desc)

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

    parser.add_argument('--productType',
                        choices=[ProductTypeMod44.PRODUCT_TYPE],
                        default=ProductTypeMod44.PRODUCT_TYPE,
                        help='Choose the product type to run.')

    parser.add_argument('-t',
                        nargs='+',
                        type=str,
                        required='True',
                        help='List of tile IDs in the form h##v## h##v## ...')

    parser.add_argument('-y',
                        type=int,
                        required='True',
                        help='The year to run')

    parser.add_argument('--workers',
                        type=int,
                        default=1,
                        help='Number of processes building the composites ' +
                             'of a band (default 1: serial).  Results are ' +
                             'the same for any number.')


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

    parser.add_argument('--writeIntermediates',
                        action='store_true',
                        help='Also write the daily inputs to 1-Days (for '
                             'diagnostics).  By default they are kept in '
                             'memory only; composites and metrics are '
                             'always written.  Results are the same.')

    args = parser.parse_args()

    BandDayFile.setWriteDailies(args.writeIntermediates)
    
    # ---
    # Low-sun mode (passed to composite workers through the environment).
    # ---
    LowSun.configure(args.lowSun,
                     None if args.lowSunCheck is None
                     else args.lowSunCheck == 'on')

    # ---
    # Logging
    # ---
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # ---
    # Product type
    #
    # This can be accomplished with registration, instead of this hard coding.
    # ---
    prodType = ProductTypeMod44(MOD44_DIR)
    
    # ---
    # Composites are built in this many processes.
    # ---
    Metrics.compositeWorkers = max(1, args.workers)

    # ---
    # Make metrics
    # ---
    for tid in args.t:

        mm = Metrics(tid, 
                     args.y, 
                     prodType, 
                     args.o, 
                     logger=logger)
    
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
