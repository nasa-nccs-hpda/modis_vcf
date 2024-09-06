#!/usr/bin/python

import argparse
from pathlib import Path
import sys

from modis_vcf.model.CompositeDayFile import CompositeDayFile
from modis_vcf.model.ProductTypeMod09G import ProductTypeMod09G


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/writeGqComposites.py -i /explore/nobackup/projects/ilab/data/MODIS/MOD09G_greenland -o /explore/nobackup/people/rlgill/SystemTesting/mfTest --numDays 10 -y 2000 -t h15v02 --startDay 213
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application create composite tifs of MODIS data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-i',
                        type=Path,
                        required='True',
                        help='Input directory for finding MODIS files')

    parser.add_argument('--startDay',
                        type=int,
                        required='True',
                        help='Julian day of the first day in the composite')

    parser.add_argument('--numDays',
                        type=int,
                        required='True',
                        help='Number of days to include in the composite')

    parser.add_argument('-o',
                        type=Path,
                        default='.',
                        help='Output directory for writing metrics')

    parser.add_argument('-t',
                        type=str,
                        required='True',
                        help='Tile ID in the form h##v##')

    parser.add_argument('-y',
                        type=int,
                        required='True',
                        help='The year to run')

    args = parser.parse_args()
    
    # ---
    # ProductType
    # ---
    gaDir = args.i / (ProductTypeMod09G.PRODUCT_TYPE + ProductTypeMod09G.GA)
    gqDir = args.i / (ProductTypeMod09G.PRODUCT_TYPE + ProductTypeMod09G.GQ)
    pt = ProductTypeMod09G(gaDir, gqDir)

    # ---
    # CompositeDayFile
    # ---
    bands = [ProductTypeMod09G.BAND1, ProductTypeMod09G.BAND2]
    
    for band in bands:
        
        cdf = CompositeDayFile(pt, 
                               args.t, 
                               args.y, 
                               args.startDay,
                               band, 
                               args.o, 
                               daysInComposite = args.numDays)
    
        cdf.toTif()


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
