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
# modis_vcf/view/makeMetricsCLV.py --productType MOD44 -o /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles/MOD44C -y 2019 -t h24v02 -m metricBandReflMin
#
# modis_vcf/view/makeMetricsCLV.py --productType MOD44 -o /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles/MOD44C -y 2019 -t h08v04 h08v05 h09v04 h09v05 h10v04 h10v05 h10v06 h11v02 h11v03 h11v04 h11v05 h11v08 h11v09 h11v10 h12v01 h12v02 h12v03 h12v04 h12v05 h12v09 h12v10 h12v12 h13v01 h13v02 h13v10 h13v11 h13v12 h16v01 h17v05 h18v03 h18v04 h18v07 
#
# modis_vcf/view/makeMetricsCLV.py --productType MOD44 -o /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles/MOD44C -y 2019 -t h19v04 h19v07 h19v08 h19v09 h19v10 h19v11 h19v12 h20v02 h20v03 h20v04 h20v06 h20v08 h20v09 h20v10 h20v11 h21v01 h21v02 h21v04 h21v05 h21v06 h21v10 h22v03 h22v04 h23v02 h23v03 h24v02 h24v03 h26v06 h27v04 h27v07 h28v11 h29v11 h29v12 h30v12 h31v11
#
# TODO: product type and input directories depend on each other, and should not
#       be hard coded
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to compute metrics.'
    parser = argparse.ArgumentParser(description=desc)

    inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')
    inDir09 = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD09A1')

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
                        choices=[ProductTypeMod09A.PRODUCT_TYPE, 
                                 ProductTypeMod44.PRODUCT_TYPE],
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
    # Product type
    #
    # This can be accomplished with registration, instead of this hard coding.
    # ---
    inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')
    inDir09 = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD09A1')
    prodType = None

    if args.productType == ProductTypeMod09A.PRODUCT_TYPE:
        
        prodType = ProductTypeMod09A(inDir09, inDir)
        
    else:
        prodType = ProductTypeMod44(inDir)
    
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
