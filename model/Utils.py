
from collections import namedtuple
from pathlib import Path
import logging

import numpy as np

from osgeo import gdal
from osgeo import gdal_array
from osgeo.osr import SpatialReference


# ----------------------------------------------------------------------------
# Class Utils
#
# This has just about become a replacement for what should be a Band class.
# ----------------------------------------------------------------------------
class Utils(object):
    
    NO_DATA = -10001
    ROWS = 4800
    COLS = 4800
    NUMPY_DTYPE = np.int16
    
    # ---
    # Cube is an ndarray.  
    # DayXref is a dictionary of {yyyydd: index} into the cube.
    # ---
    Band = namedtuple('Band', 'name, dataType, cube, dayXref')

    modisSinusoidal = SpatialReference()
    
    modisSinusoidal.ImportFromProj4(
        '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84' + 
        ' +datum=WGS84 +units=m +no_defs')
    
    # ------------------------------------------------------------------------
    # createDsFromParams
    # ------------------------------------------------------------------------
    @staticmethod
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

        ds.SetSpatialRef(Utils.modisSinusoidal)
        
        return ds
    
    # ------------------------------------------------------------------------
    # createDs
    # ------------------------------------------------------------------------
    @staticmethod
    def createDs(outName: Path, band: Band) -> gdal.Dataset:
        
        shape = band.cube.shape[1:]

        ds = Utils.createDsFromParams(outName, 
                                      shape, 
                                      band.cube.shape[0], 
                                      band.dataType)
        
        return ds
    
    # ------------------------------------------------------------------------
    # writeBand
    #
    # The "day" argument is of the form yyyyddd.
    # ------------------------------------------------------------------------
    @staticmethod
    def writeBand(outDir: Path,
                  band: Band,
                  logger: logging.RootLogger,
                  name: str = None,
                  day: str = None) -> Path:

        name = name or band.name
        outName = outDir / (name + '.tif')
        
        if logger:
            logger.info('Writing ' + str(outName))
            
        if not outName.exists():
            
            ds = Utils.createDs(outName, band)
            outBandIndex = 0
        
            daysToWrite = day or band.dayXref

            for dayKey in daysToWrite:

                cubeIndex = band.dayXref[dayKey]
                day: np.ndarray = band.cube[cubeIndex]
                outBandIndex += 1
                gdBand = ds.GetRasterBand(outBandIndex)
                gdBand.WriteArray(day)
                gdBand.SetMetadata({'Name': dayKey})
                gdBand.SetNoDataValue(Utils.NO_DATA)
                gdBand.FlushCache()
                gdBand = None

            ds = None

        return outName

    # ------------------------------------------------------------------------
    # readBand
    # ------------------------------------------------------------------------
            