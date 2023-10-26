#!/usr/bin/python

import argparse
import logging
from pathlib import Path
import sys

import numpy as np

from osgeo import gdal
from osgeo import gdal_array

from modis_vcf.model.Band import Band
from modis_vcf.model.CollateBandsByDate import CollateBandsByDate
from modis_vcf.model.Pair import Pair


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/dumpQa.py -o /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/debugQa
# -----------------------------------------------------------------------------
def main():

    desc = 'Use this application to print QA components.'
    parser = argparse.ArgumentParser(description=desc)

    inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')

    parser.add_argument('-i',
                        type=Path,
                        default=inDir,
                        help='Input directory')

    parser.add_argument('-o',
                        type=Path,
                        default=Path('.'),
                        help='Output directory')

    parser.add_argument('-y',
                        type=int,
                        default=2019,
                        help='The year to run')

    parser.add_argument('-t',
                        type=str,
                        default='h09v05',
                        help='Tile ID in the form h##v##')

    parser.add_argument('-b',
                        type=str,
                        default=Pair.BAND1,
                        help='Band name.  See Pair.py.')

    args = parser.parse_args()

    # ---
    # Set up logging.
    # ---
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # ---
    # Collect the days and dump the QA.
    # ---
    cbbd = CollateBandsByDate(args.t, args.y, args.i, args.o, logger)
    cbbd._collectInputFiles()
    
    testDays = None  # ['2019177']
    days = testDays or cbbd._pairs
    
    for key in days:
        
        print('Day:', key)
        pair = cbbd._pairs[key]
        componentsToWrite = {}

        stateName = pair._cqMate.dataset.GetSubDatasets()[Pair.STATE_INDEX][0]
        state = gdal.Open(stateName).ReadAsArray()
        
        # ---
        # State matches a third-party extraction.
        #
        # (gdalNgmt) [rlgill@ilab203 debugQa]$ diff 2019177-state.tif /explore/nobackup/people/jacaraba/ForPeople/ForMark/VCF/MOD44CQ.A2019177.h09v05.061.2020303094028-state.bin
        # ---
        # stateOutName = args.o / (key + '-state.tif')
        # print('Writing', str(stateOutName))
        # state.tofile(stateOutName)
        
        # ---
        # This works.
        # pge72
        # define cloud 0x3
        # ---
        cloud = state & 3
        componentsToWrite['cloud'] = cloud
        print('Cloud unique:', np.unique(cloud, return_counts=True))

        # ---
        # This works.
        # pge72
        # define shadow 0x4  
        # ---
        shadow = state & 4
        mask = np.where((state == 0), 0, 1).astype(np.uint16)
        componentsToWrite['shadow'] = shadow
        print('Shadow unique:', np.unique(shadow, return_counts=True))

        # ---
        # pge72
        # Adjacency is 0x2000 = 8192
        # ((outputs->state[i][j] & 28672) >> 7) | /* snow, adjacency, brdf */
        # 8192 =  0b010000000000000
        # 28672 = 0b111000000000000
        # >> 7 = 0b11100000
        # Does that mean adjaceny is the middle "1"?
        # 0b11100000 becomes
        # 0b01000000 = 64
        # ---
        adjacency = state & 8192
        componentsToWrite['adjacency'] = adjacency
        print('Adjacency unique:', np.unique(adjacency, return_counts=True))

        # ---
        # This works.
        # ---
        aerosol = (state & 192) >> 6
        mask = np.where((aerosol == 3), 1, 0).astype(np.uint16)
        componentsToWrite['aerosol'] = mask
        print('Aerosol unique:', np.unique(aerosol, return_counts=True))

        write(componentsToWrite, key, args.o)
        
# -----------------------------------------------------------------------------
# write
# -----------------------------------------------------------------------------
def write(componentsToWrite: dict, day: str, outDir: Path) -> None:
    
    outName = outDir / (day + '.tif')
    print('Writing ' + str(outName))
    
    sampleComponent = list(componentsToWrite.values())[0]
    
    # dataType = \
    #     gdal_array.NumericTypeCodeToGDALTypeCode(sampleComponent.dtype)

    dataType = gdal.GDT_UInt16

    ds = gdal.GetDriverByName('GTiff').Create(
        str(outName),
        sampleComponent.shape[0],
        sampleComponent.shape[1],
        len(componentsToWrite),
        dataType,
        options=['COMPRESS=LZW', 'BIGTIFF=YES'])

    ds.SetSpatialRef(Band.modisSinusoidal)
    outBandIndex = 0
    
    for comp in componentsToWrite:
        
        outBandIndex += 1
        gdBand = ds.GetRasterBand(outBandIndex)
        value = componentsToWrite[comp]
        gdBand.WriteArray(value)
        # gdBand.SetNoDataValue(Band.NO_DATA)
        gdBand.SetMetadata({'Name': comp})
        gdBand.FlushCache()
        gdBand = None

    ds = None
    

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
    
        
    
