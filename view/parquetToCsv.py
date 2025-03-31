#!/usr/bin/python

import argparse
from pathlib import Path
import sys

import pandas as pd


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/parquetToCsv.py -i /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/h09v04-2019.parq
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to convert Parquet files to CSV.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-i',
                        type=Path,
                        help='Input Parquet file to convert to CSV')

    parser.add_argument('--stats',
                        default=False,
                        action='store_true',
                        help='Print statistics about the Parquet file')

    args = parser.parse_args()

    print('Reading ' + str(args.i))
    parq = pd.read_parquet(args.i)
    
    outFile = args.i.with_suffix('.csv') 
    print('Writing ' + str(outFile))
    parq.to_csv(outFile)
    

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
