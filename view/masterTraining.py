#!/usr/bin/python

import argparse
import logging
from pathlib import Path
import sys

from modis_vcf.model.MasterTraining import MasterTraining


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/masterTraining.py -t /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles/MOD44C/training
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to build the master training file.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-t',
                        type=Path,
                        required='True',
                        help='Input training directory')

    args = parser.parse_args()

    # ---
    # Logging
    # ---
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    
    mt = MasterTraining(args.t, logger)
    mt.writeCsv(args.t)
    

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
