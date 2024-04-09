#!/usr/bin/python

import argparse
import logging
from pathlib import Path
import sys

from modis_vcf.model.CollateBandsByDate import CollateBandsByDate


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/collateOneBand.py -o /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles -t h24v04 -y 2019 -b BAND_1
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to compute metrics.'
    parser = argparse.ArgumentParser(description=desc)

    inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')

    parser.add_argument('-b',
                        type=str,
                        required='True',
                        help='Band name.  See Pair.py.')

    parser.add_argument('-i',
                        type=Path,
                        default=inDir,
                        help='Input directory for finding MODIS files')

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
    # Logging
    # ---
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    cbbd = CollateBandsByDate(args.t, 
                              args.y, 
                              args.i, 
                              args.o, 
                              logger, 
                              write=True)
                           
    cbbd.getBand(args.b)
    

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
