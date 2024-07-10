#!/usr/bin/python

import argparse
import logging
from pathlib import Path
import sys

from modis_vcf.model.MaskMod44b import MaskMod44b


# ----------------------------------------------------------------------------
# main
#
# python modis_vcf/view/maskMod44bCLV.py \
#  -hdf /css/modis/Collection6/L3/MOD44B-VCF/2019/065 \
#  -mask /panfs/ccds02/nobackup/people/mfrost2/temp/Coll6_0s_tifs \
#  -year 2019 \
#  -tile h10v02 \
#  -subdataset Percent_Tree_Cover \
#  -o . \
# ----------------------------------------------------------------------------
def main():

    desc = 'Use this application to create masked MOD44B HDF files'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-tile',
                        type=str,
                        required=True,
                        help='MODIS Tile string. E.g. h09v05')

    parser.add_argument('-hdf',
                        type=Path,
                        required=True,
                        help='Directory containing MODIS MOD44B HDF files')

    parser.add_argument('-mask',
                        type=Path,
                        required=True,
                        help='Directory containing MODIS MOD44B mask files')

    parser.add_argument('-year',
                        type=int,
                        required=True,
                        help='The year to run')

    parser.add_argument('-subdataset',
                        type=str,
                        default='Percent_Tree_Cover',
                        help='Subdataset name to mask')

    parser.add_argument('-out',
                        type=Path,
                        default=Path('.'),
                        help='Output directory')

    args = parser.parse_args()

    # ---
    # logging
    # ---
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    maskMod44b = MaskMod44b(args.tile, args.year, args.hdf,
                            args.mask, args.subdataset, args.out, logger)
    maskMod44b.run()


# ----------------------------------------------------------------------------
# Invoke the main
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
