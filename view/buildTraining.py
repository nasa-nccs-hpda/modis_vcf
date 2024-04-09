#!/usr/bin/python

import argparse
import glob
import logging
from pathlib import Path
import sys

from modis_vcf.model.BuildTraining import BuildTraining


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/buildTraining.py -o /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles -m metricWarmest3MeanBandRefl -t h09v04 h09v05
#
# modis_vcf/view/buildTraining.py -o /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles -y 2019 -t h08v04  h11v04  h12v05  h17v05  h19v12  h21v02  h24v04 h08v05  h11v05  h12v09  h18v03  h20v02  h21v04  h26v06 h09v04  h11v08  h12v10  h18v04  h20v03  h21v05  h27v04 h09v05  h11v09  h12v12  h18v07  h20v06  h21v06  h27v06 h10v04  h11v10  h13v01  h19v04  h20v08  h21v10  h27v07 h10v05  h12v01  h13v10  h19v08  h20v09  h22v03  h29v11 h10v06  h12v02  h13v11  h19v09  h20v10  h23v02  h29v12 h11v02  h12v03  h13v12  h19v10  h20v11  h23v03  h30v12 h11v03  h12v04  h16v01  h19v11  h21v01  h24v03  h31v11

# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to build training data.'
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

    parser.add_argument('-n',
                        type=str,
                        default='PercentTree',
                        help='Name of the training data')

    parser.add_argument('-o',
                        type=Path,
                        default='.',
                        help='Output directory for writing metrics')

    parser.add_argument('--stats',
                        default=False,
                        action='store_true',
                        help='Print statistics about the Parquet file')

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
    
    # ---
    # Logging
    # ---
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    
    # Get the tile IDs of the sample data.
    bt = BuildTraining(args.y, args.i, args.o, logger, args.n, args.t, args.m)
    bt.run()
    
    if args.stats:
        bt.statistics()
        

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
