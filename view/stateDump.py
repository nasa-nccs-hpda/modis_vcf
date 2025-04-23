#!/usr/bin/python

import argparse
from pathlib import Path
import sys

import numpy as np

from osgeo import gdal


# -----------------------------------------------------------------------------
# main
#
# python modis_vcf/view/stateDump.py -i /css/modis/Collection6.1/L3/MOD44B-VCF/dev/2019/MOD44CQ.A2019353.h31v11.061.2020323111251.hdf -o /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/SystemTests/stateDump
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
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to write a Geotiff for each MODIS \
           state field.'
           
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-i',
                        required=True,
                        type=Path,
                        help='Full path to MOD44 file.')

    parser.add_argument('-o',
                        required=True,
                        type=Path,
                        default='.',
                        help='Output directory')

    args = parser.parse_args()

    # Ensure it is a CQ file.
    inFile = Path(args.i)

    if not inFile.stem.split('.')[0].endswith('CQ'):
        raise RuntimeError('Input file must be a CQ file.')
        
    # Validate output directory.
    if not args.o.exists():
        raise RuntimeError('The output directory does not exist.')
    
    # Read.
    print('Reading', args.i)
    ds: gdal.Dataset = gdal.Open(args.i)
    stateDs = gdal.Open(ds.GetSubDatasets()[1][0])  # State is the first band.
    state: np.ndarray = stateDs.ReadAsArray(buf_xsize=4800, buf_ysize=4800)
    
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

    # Range checking.
    assert(cloud.min() == 0)
    assert(cloud.max() <= 3)
    assert(shadow.min() == 0)
    assert(shadow.max() <= 1)
    assert(land.min() == 0)
    assert(land.max() <= 7)
    assert(aerosol.min() == 0)
    assert(aerosol.max() <= 3)
    assert(cirrus.min() == 0)
    assert(cirrus.max() <= 3)
    assert(intCloud.min() == 0)
    assert(intCloud.max() <= 1)
    assert(fire.min() == 0)
    assert(fire.max() <= 1)
    assert(snow.min() == 0)
    assert(snow.max() <= 1)
    assert(adjacency.min() == 0)
    assert(adjacency.max() <= 1)
    assert(brdf.min() == 0)
    assert(brdf.max() <= 1)
    assert(snowAlgo.min() == 0)
    assert(snowAlgo.max() <= 1)

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
              'snowAlgo': snowAlgo}
    
    # Write.
    for field in fields:
        
        outName: Path = args.o / (inFile.stem + '-' + field + '.tif')
        print('Writing', outName)
        
        outDs = gdal.GetDriverByName('GTiff').Create(
            str(outName), 4800, 4800, 1, options=['COMPRESS=LZW'])
        
        gdBand = outDs.GetRasterBand(1)
        gdBand.WriteArray(fields[field].astype(np.uint8))
    

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
