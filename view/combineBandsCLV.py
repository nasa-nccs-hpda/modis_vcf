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
# modis_vcf/view/combineBandsCLV.py -o /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to compute metrics.'
    parser = argparse.ArgumentParser(description=desc)

    inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')

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
                        help='Tile ID in the form h##v##')

    parser.add_argument('-y',
                        type=int,
                        help='The year to run')

    args = parser.parse_args()
    inDir = args.i
    inYear = str(args.y or '*')  # Need to 'or' before 'str'.
    inTile = args.t
    
    # ---
    # Logging
    # ---
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # ---
    # Which tiles?
    #
    # Only consider day 65, the first day of the VCF year, and Metrics will
    # find the rest of the year's files.
    #
    # MOD44CH.A2020049.h21v02.061.2020335045123.hdf
    # ---
    globStr = 'MOD44C'
    if inYear: globStr += '*A' + inYear + '065'
    if inTile: globStr += '*' + inTile
    globStr += '*.hdf'
    
    hdfs = inDir.glob(globStr)
    yearTiles = []
    
    for hdf in hdfs:
        
        splits = hdf.name.split('.')
        year = int(splits[1][1:5])
        tile = splits[2]
        yearTiles.append((year, tile))
        
    # ---
    # Create combined bands.
    # ---
    incompleteTiles = []
    
    for year, tile in set(yearTiles):
        
        logger.info('Processing ' + tile)
        
        try:
            mm = Metrics(tile, year, inDir, args.o, logger=logger)
    
            for band in Pair.BANDS:
                mm.getBand(band)

        except RuntimeError as e:
            
            incompleteTiles.append(tile)
            logger.error(e)
            
    logger.warning('These incomplete tiles were skipped: ' +
                   str(incompleteTiles))
            

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
