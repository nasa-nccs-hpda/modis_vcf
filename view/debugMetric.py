#!/usr/bin/python

import argparse
import glob
# import logging
from pathlib import Path
import sys

import numpy as np

from osgeo import gdal

from modis_vcf.model.MakeMetrics import MakeMetrics
from modis_vcf.model.Mate import Mate
from modis_vcf.model.Pair import Pair
from modis_vcf.model.Utils import Utils


# -------------------------------------------------------------------------
# dumpBand
# -------------------------------------------------------------------------
def dumpBand() -> None:

# -------------------------------------------------------------------------
# write
# -------------------------------------------------------------------------
def write(outName: Path, band: np.ndarray, dataType: int) -> None:
    
    print('Writing', outName)
    ds = Utils.createDsFromParams(outName, band.shape, 1, dataType)
    gdBand = ds.GetRasterBand(1)
    gdBand.WriteArray(band)
    gdBand.SetNoDataValue(Utils.NO_DATA)
    gdBand.FlushCache()
    gdBand = None
    ds = None
    
# -----------------------------------------------------------------------------
# main
#
# modis_vcf/view/debugMetric.py -o /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/debug353
# -----------------------------------------------------------------------------
def main():

    desc = 'Use this application to debug a metric.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--tileID',
                        type=str,
                        default='h09v05',
                        help='Tile ID in the form h##v##')

    parser.add_argument('--year',
                        type=int,
                        default=2019,
                        help='The year to run')

    inDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')

    parser.add_argument('-i',
                        default=inDir,
                        help='Input directory')

    parser.add_argument('-o',
                        type=Path,
                        default=Path('.'),
                        help='Output directory')

    parser.add_argument('-b',
                        type=str,
                        default=Pair.BAND1,
                        help='Band name.  See Pair.py.')

    args = parser.parse_args()

    bandName = args.b

    # ---
    # Get the 353 day because we are told that is a good day to test.
    # There should be one pair.
    # ---
    globStr = '*A' + str(args.year) + '353' + '.' + args.tileID + '.*.hdf'
    hdfFiles = list(args.i.glob(globStr))
    
    try:
        for hdfFile in hdfFiles:
            linkPath = args.o / hdfFile.name
            linkPath.symlink_to(hdfFile)

    except FileExistsError:
        pass
        
    # ---
    # Instantiate the pair.
    # ---
    chMate = None
    cqMate = None
    
    for hdfFile in hdfFiles:
        
        if 'MOD44CH' in hdfFile.name:
            
            chMate = Mate(hdfFile)
            
        elif 'MOD44CQ' in hdfFile.name:
            
            cqMate = Mate(hdfFile)
            
        else:
            raise RuntimeError('Found unrecognized mate ' + str(hdfFile))
            
    pair = Pair(chMate, cqMate)
    
    # ---
    # Read the mate directly.  Usually, Pair does this.
    # ---
    print('--- MATE ---')
    mate, index = pair._getMate(bandName)
    mateBand, dataType = mate.read(index)
    print('Num no-data values:', np.count_nonzero(mateBand == Utils.NO_DATA))
    outName = args.o / 'mate.tif'
    write(outName, mateBand, dataType)
    
    # ---
    # Bands are usually read from Pair.read(), which consider other factors
    # unknown to a mate.
    # ---
    print('--- PAIR NO QA ---')
    pairBandNoQa, dataType = pair.read(bandName)
    
    print('Num no-data values:', 
          np.count_nonzero(pairBandNoQa == Utils.NO_DATA))

    print('Number of clamps found:', np.count_nonzero(pairBandNoQa == 16000))
    outName = args.o / 'pairBandNoQa.tif'
    write(outName, pairBandNoQa, dataType)
    
    # ---
    # Pair.read() clamps based on solar zenith.
    # ---
    print('--- SOLAR ZENITH ---')
    solz = pair.solarZenith
    print('Num no-data values:', np.count_nonzero(solz == Utils.NO_DATA))
    print('Zenith cut off:', Pair.ZENITH_CUTOFF)
    numClampsExp = np.count_nonzero(solz >= Pair.ZENITH_CUTOFF)
    print('Number of clamps expected:', numClampsExp)
    print('Number of clamps found:', np.count_nonzero(pairBandNoQa == 16000))
    outName = args.o / 'solz.tif'
    dataType = pair._cqMate.read(Pair.SOLAR_ZENITH_INDEX)[1]
    write(outName, mateBand, dataType)
    
    # ---
    # Now read it and apply the QA mask.
    # ---
    print('--- STATE ---')
    
    state = gdal.Open(pair._cqMate.dataset.GetSubDatasets() \
        [Pair.STATE_INDEX][0]).ReadAsArray().astype(np.int16)

    print('Num no-data values:', np.count_nonzero(state == Utils.NO_DATA))

    outName = args.o / 'state.tif'
    write(outName, state, state.dtype)
        
    # print('--- QA ---')
    # import pdb
    # pdb.set_trace()
    #
    # gdalDataType = gdal.Open(pair._cqMate.dataset.GetSubDatasets() \
    #                [Pair.STATE_INDEX][0]).GetRasterBand(1).DataType
    #
    # print('GDAL data type:', gdalDataType)  # 1 is GDT_Byte
    #
    # outName = args.o / 'qa.tif'
    # qa = pair.qaMask
    # print('QA dtype', qa.dtype)
    # write(outName, qa, qa.dtype)  # Invalid type np int64
    
    print('--- PAIR WITH QA ---')
    pairBandQa, dataType = pair.read(bandName, applyQa=True)
    print('Num no-data values:', np.count_nonzero(pairBandQa == Utils.NO_DATA))
    outName = args.o / 'pairBandQa.tif'
    write(outName, pairBandNoQa, dataType)
    print('Number of clamps found:', np.count_nonzero(pairBandQa == 16000))
    
    # ---
    # Metric 1 is easy.
    # ---
    print('--- METRIC1 ---')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    mm = MakeMetrics(args.tileID, args.year, args.i, logger=logger)
    
    # Print the combined information.
    
    # Run the metric.
    b1 = mm.getBand(Pair.BAND1)  # This combines periods.
    mm.metric1()
    
    
# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
    
    
    
    
