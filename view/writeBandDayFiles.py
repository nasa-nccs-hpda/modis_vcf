#!/usr/bin/python

import argparse
import logging
from pathlib import Path
import sys


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/writeBandDayFiles.py -i /explore/nobackup/projects/ilab/data/MODIS/MOD44C -a /explore/nobackup/projects/ilab/data/MODIS/MOD09A1 -o /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles/MOD44C -t h09v05 -y 2019 -p MOD09
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to write bands for individual days.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-a',
                        type=Path,
                        help='Auxilliary directory for finding MODIS files')

    parser.add_argument('-i',
                        type=Path,
                        help='Input directory for finding MODIS files')

    parser.add_argument('-o',
                        type=Path,
                        default='.',
                        help='Output directory for writing metrics')

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
    # ProductType
    # ---

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
