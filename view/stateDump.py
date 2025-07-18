#!/usr/bin/python

import argparse
from pathlib import Path
import sys

import numpy as np

from osgeo import gdal


# -----------------------------------------------------------------------------
# main
#
# python modis_vcf/view/stateDump.py -i /css/modis/Collection6.1/L3/MOD44B-VCF/dev/2010/MOD44CQ.A2010209.h12v02.061.2021169001105.hdf -o /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/SystemTests/stateDump
#
# python modis_vcf/view/stateDump.py -d /css/modis/Collection6.1/L3/MOD44B-VCF/dev/2010 -o /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/SystemTests/stateDump --noWrite
# -----------------------------------------------------------------------------
def main():

    desc = 'Use this application to interact with each MODIS state field.'

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-o',
                        required=True,
                        type=Path,
                        default='.',
                        help='Output directory')

    parser.add_argument('--noWrite',
                        action='store_true',
                        default=False,
                        help='Do not write Geotiffs to output directory.')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-d',
                       type=Path,
                       help='Directory containing MOD44 file.')

    group.add_argument('-i',
                       type=Path,
                       help='Full path to MOD44 file.')

    args = parser.parse_args()
    
    # Validate output directory.
    if not args.o.exists():
        raise RuntimeError('The output directory does not exist.')

    files = []

    if args.i:
        
        # Ensure it is a CQ file.
        inFile = Path(args.i)

        if not inFile.stem.split('.')[0].endswith('CQ'):
            raise RuntimeError('Input file must be a CQ file.')
            
        files.append(inFile)
            
    elif args.d:
        
        files = args.d.glob('*CQ*.hdf')

    # Read.
    for f in files:
        
        fields = read(f)
        rangeCheck(fields)

        # Screening queries
        query(fields)

        # Write.
        if not args.noWrite:
            write(fields, args.o, inFile)

# -----------------------------------------------------------------------------
# query
# -----------------------------------------------------------------------------
def query(fields: dict) -> None:
    
    print('             Cloud set:', (fields['cloud'] > 0).any())
    print('              Snow set:', (fields['snow'] > 0).any())
    print('    Snow algorithm set:', (fields['snowAlgo'] > 0).any())
    
    print('Snow and snow algo set:', 
          np.logical_and(fields['snow'] > 0, fields['snowAlgo'] > 0).any())

    print('    Snow and cloud set:', 
          np.logical_and(fields['snow'] > 0, fields['cloud'] > 0).any())
    
# -----------------------------------------------------------------------------
# rangeCheck
# -----------------------------------------------------------------------------
def rangeCheck(fields: dict) -> None:

    assert fields['cloud'].min() == 0
    assert fields['cloud'].max() <= 3
    assert fields['shadow'].min() == 0
    assert fields['shadow'].max() <= 1
    assert fields['land'].min() == 0
    assert fields['land'].max() <= 7
    assert fields['aerosol'].min() == 0
    assert fields['aerosol'].max() <= 3
    assert fields['cirrus'].min() == 0
    assert fields['cirrus'].max() <= 3
    assert fields['intCloud'].min() == 0
    assert fields['intCloud'].max() <= 1
    assert fields['fire'].min() == 0
    assert fields['fire'].max() <= 1
    assert fields['snow'].min() == 0
    assert fields['snow'].max() <= 1
    assert fields['adjacency'].min() == 0
    assert fields['adjacency'].max() <= 1
    assert fields['brdf'].min() == 0
    assert fields['brdf'].max() <= 1
    assert fields['snowAlgo'].min() == 0
    assert fields['snowAlgo'].max() <= 1

# -----------------------------------------------------------------------------
# read
# -----------------------------------------------------------------------------
def read(inFile: Path) -> dict:
    
    print('Reading', inFile)
    gdal.UseExceptions()
    ds: gdal.Dataset = gdal.Open(str(inFile))
    stateDs = gdal.Open(ds.GetSubDatasets()[0][0])
    state: np.ndarray = stateDs.ReadAsArray(buf_xsize=4800, buf_ysize=4800)

    # ---
    # define snow_algorithm 0x8000      32768
    # define BRDF_correction 0x4000     16384
    # define cloud_adjacency 0x2000      8192
    # define snow 0x1000                 4096
    # define fire 0x800                  2048
    # define internal_cloud 0x400        1024
    # define cirrus 0x300                 768
    # define aerosol 0xC0                 192
    # define land 0x38                     56
    # define shadow 0x4                     4
    # define cloud 0x3                      3
    # define not_internal_cloud 0xFBFF  64511
    # 
    # 1111111000000000
    # 5432109876543210
    # 1000000000000000 snow algorithm
    #  100000000000000 BRDF correction
    #   10000000000000 adjacency
    #    1000000000000 snow
    #     100000000000 fire
    #      10000000000 internal cloud
    #       1100000000 cirrus
    #         11000000 aerosol
    #           111000 land
    #              100 shadow
    #               11 cloud
    #
    # 0b  100110000000
    # ---
    cloud = state & 3
    shadow = (state & 4) >> 2
    land = (state & 56) >> 3
    aerosol = (state & 192) >> 6
    cirrus = (state & 768) >> 8
    intCloud = (state & 1024) >> 10
    fire = (state & 2048) >> 11
    snow = (state & 4096) >> 12
    adjacency = (state & 8192) >> 13
    brdf = (state & 16384) >> 14
    snowAlgo = (state & 32768) >> 15

    cloud2 = state & 3
    shadow2 = (state >> 2) & 1
    land2 = (state >> 3) & 7
    aerosol2 = (state >> 6) & 3
    cirrus2 = (state >> 8) & 3
    intCloud2 = (state >> 10) & 1
    fire2 = (state >> 11) & 1
    snow2 = (state >> 12) & 1
    adjacency2 = (state >> 13) & 1
    brdf2 = (state >> 14) & 1
    snowAlgo2 = (state >> 15) & 1

    # if cloud != cloud2 or \
    #     shadow != shadow2 or \
    #     land != land2 or \
    #     aerosol != aerosol2 or \
    #     cirrus != cirrus2 or \
    #     intCloud != intCloud2 or \
    #     fire != fire2 or \
    #     snow != snow2 or \
    #     adjacency != adjacency2 or \
    #     brdf != brdf2 or \
    #     snowAlgo != snowAlgo2:
    #
    #     raise RuntimeError('Field extractions do not match.')

    # VCF masking
    solzDs = gdal.Open(ds.GetSubDatasets()[2][0])
    solz: np.ndarray = solzDs.ReadAsArray(buf_xsize=4800, buf_ysize=4800)
    zenithCutOff = 72

    mask = np.where((cloud == 0) &
                    (shadow == 0) &
                    (aerosol != 3) &
                    (adjacency == 0) &
                    (solz > 0) &
                    (solz < zenithCutOff),
                    0,
                    1).astype(np.uint8)

    fields = {'cloud': cloud,
              'shadow': shadow,
              'land': land,
              'aerosol': aerosol,
              'cirrus': cirrus,
              'intCloud': intCloud,
              'fire': fire,
              'snow': snow,
              'adjacency': adjacency,
              'brdf': brdf,
              'snowAlgo': snowAlgo,
              'mask': mask}
              
    return fields

# -----------------------------------------------------------------------------
# write
# -----------------------------------------------------------------------------
def write(fields: dict, outDir: Path, inFile: Path) -> None:

    for field in fields:

        outName: Path = outDir / (inFile.stem + '-' + field + '.tif')
        print('Writing', outName)

        outDs = gdal.GetDriverByName('GTiff').Create(
            str(outName), 
            4800, 
            4800, 
            options=['COMPRESS=LZW', 'INTERLEAVE=PIXEL'])

        outDs.WriteArray(fields[field])

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
