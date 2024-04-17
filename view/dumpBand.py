#!/usr/bin/python

import argparse
import logging
from pathlib import Path
import sys
import tarfile

import numpy as np

from osgeo import gdal
from osgeo import gdal_array

from modis_vcf.model.Band import Band
from modis_vcf.model.CollateBandsByDate import CollateBandsByDate
from modis_vcf.model.MakeMetrics import MakeMetrics
from modis_vcf.model.Mate import Mate
from modis_vcf.model.Pair import Pair


# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/dumpBand.py -o /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/debug353
# -----------------------------------------------------------------------------
def main():

    desc = 'Use this application to dump band information.'
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
    # Collect the days and dump them.
    # ---
    cbbd = CollateBandsByDate(args.t, args.y, args.i, args.o, logger)
    band = cbbd.getBand(args.b)
    filesToTar = []
    
    for day in band.dayXref:
        
        pair = cbbd.pairs[day]
        dayFiles = dumpDay(pair, args.b, day, args.o, 'RawBand')
        filesToTar += dayFiles
        
    # ---
    # Dump the combined days.
    # ---
    mm = MakeMetrics(args.t, args.y, args.i, args.o, logger=logger)
    comboBand = mm.getBand(args.b)

    for day in comboBand.dayXref:

        index = comboBand.dayXref[day]
        dayBand = comboBand.cube[index, :, :]
        outName = args.o / ('Combo-' + args.b + '-' + day + '.tif')
        filesToTar.append(outName)
        write(outName, dayBand, comboBand.dataType)

    tarFile = args.o / ('Dump-' + args.b + '.tar')
    print('Writing', tarFile)

    with tarfile.open(tarFile, 'w') as tar:
        for name in filesToTar:
            tar.add(name, arcname=name.name)
            
# -----------------------------------------------------------------------------
# dumpDay
# -----------------------------------------------------------------------------
def dumpDay(pair: Pair, 
            bandName: str, 
            day: str, 
            outDir: Path, 
            prefix: str) -> Path:
    
    # Header
    print('----------------------')
    mate: Mate, subDsIndex: int = pair._getMate(bandName)
    print('HDF:', mate.fileName)

    # Mate
    mateBand: np.ndarray, dataType: int = mate.read(subDsIndex)
    print('Num no-data:', np.count_nonzero(mateBand == Band.NO_DATA))
    mateName = outDir / (prefix + '-' + bandName + '-' + day + '-mate.tif')
    write(mateName, mateBand, dataType)

    # Pair without QA
    pairBandNoQa: np.ndarray, dataType: int = pair.read(bandName)
    print('Num no-data:', np.count_nonzero(pairBandNoQa == Band.NO_DATA))
    print('Number of clamps found:', np.count_nonzero(pairBandNoQa == 16000))
    
    outName = outDir / \
        (prefix + '-' + bandName + '-' + day + '-pairBandNoQa.tif')
        
    write(outName, pairBandNoQa, dataType)

    # Solar zenith
    solz: np.ndarray = pair.solarZenith
    print('Num no-data values:', np.count_nonzero(solz == Band.NO_DATA))
    print('Zenith cut off:', Pair.ZENITH_CUTOFF)
    numClampsExp = np.count_nonzero(solz >= Pair.ZENITH_CUTOFF)
    print('Number of clamps expected:', numClampsExp)
    print('Number of clamps found:', np.count_nonzero(pairBandNoQa == 16000))
    outName = outDir / (prefix + '-' + bandName + '-' + day + '-solz.tif')
    dataType = pair._cqMate.read(Pair.SOLAR_ZENITH_INDEX)[1]
    write(outName, mateBand, dataType)
    
    # State
    state = gdal.Open(pair._cqMate.dataset.GetSubDatasets() \
        [Pair.STATE_INDEX][0]).ReadAsArray().astype(np.int16)

    print('Num no-data values:', np.count_nonzero(state == Band.NO_DATA))
    outName = outDir / (prefix + '-' + bandName + '-' + day + '-state.tif')
    write(outName, state, state.dtype)

    # QA mask
    print('QA unique values:', np.unique(pair.qaMask))
    outName = outDir / (prefix + '-' + bandName + '-' + day + '-qa.tif')
    dataType = gdal_array.NumericTypeCodeToGDALTypeCode(pair.qaMask.dtype)
    write(outName, pair.qaMask, dataType)
    
    # Pair with QA
    pairBandQa, dataType = pair.read(bandName, applyQa=True)
    print('Num no-data values:', np.count_nonzero(pairBandQa == Band.NO_DATA))

    pairBandQaName = outDir / \
        (prefix + '-' + bandName + '-' + day + '-pairBandQa.tif')

    write(pairBandQaName, pairBandQa, dataType)
    print('Number of clamps found:', np.count_nonzero(pairBandQa == 16000))
    
    return mateName, pairBandQaName
    
# -----------------------------------------------------------------------------
# write
# -----------------------------------------------------------------------------
def write(outName: Path, band: np.ndarray, dataType: int) -> None:
    
    print('Writing', outName)
    ds = createDsFromParams(outName, band.shape, 1, dataType)
    gdBand = ds.GetRasterBand(1)
    gdBand.WriteArray(band)
    gdBand.SetNoDataValue(Band.NO_DATA)
    gdBand.FlushCache()
    gdBand = None
    ds = None

# ------------------------------------------------------------------------
# createDsFromParams
# ------------------------------------------------------------------------
def createDsFromParams(outName: Path, 
                       shape: tuple, 
                       numBands: int,
                       inDataType: int) -> gdal.Dataset:
    
    # Translate Numpy data types to GDAL types.  
    dataType = inDataType
    
    try:
        
        dataType = gdal_array.NumericTypeCodeToGDALTypeCode(inDataType)
    
    except TypeError:
        pass
    
    ds = gdal.GetDriverByName('GTiff').Create(
        str(outName),
        shape[1],
        shape[0],
        numBands,
        dataType,
        options=['COMPRESS=LZW'])

    ds.SetSpatialRef(Band.modisSinusoidal)
    
    return ds
   
    
# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
    
