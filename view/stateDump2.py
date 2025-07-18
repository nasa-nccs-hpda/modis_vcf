#!/usr/bin/python

import argparse
from pathlib import Path
import sys

import numpy as np

from osgeo import gdal


# -----------------------------------------------------------------------------
# main
#
# python modis_vcf/view/stateDump.py 2112
# -----------------------------------------------------------------------------
def main():

    desc = 'Use this application interpret a state value.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('value', type=int)
    args = parser.parse_args()
    
    print('First two lines show each columns bit number')
    print('1111111000000000')
    print('5432109876543210')
    print('1000000000000000 snow algorithm')
    print(' 100000000000000 BRDF correction')
    print('  10000000000000 adjacency')
    print('   1000000000000 snow')
    print('    100000000000 fire')
    print('     10000000000 internal cloud')
    print('      1100000000 cirrus')
    print('        11000000 aerosol')
    print('          111000 land')
    print('             100 shadow')
    print('              11 cloud')
    print(bin(args.value)[2:].zfill(16))


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
