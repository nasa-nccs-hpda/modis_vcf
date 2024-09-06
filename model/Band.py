
import logging
from pathlib import Path

import numpy as np

from osgeo import gdal
from osgeo import gdal_array
from osgeo.osr import SpatialReference

from core.model.GeospatialImageFile import GeospatialImageFile


# ----------------------------------------------------------------------------
# Class Band
#
# Possibly convert to a Generic class, so it can properly handle float and 
# integer data types in the cubes, namely NaN (float) and no-data (integer).
# https://medium.com/@steveYeah/using-generics-in-python-99010e5056eb
#
# TODO: Is dayXref really, bandXref?  This class originally represented a
#       different abstraction, and was improperly exploited to hold metrics.
# ----------------------------------------------------------------------------
class Band(object):
    
    NO_DATA = -10001
    ROWS = 4800
    COLS = 4800
    NUMPY_DTYPE = np.int16
    METADATA_BAND_NAME = 'Name'

    modisSinusoidal = SpatialReference()
    
    modisSinusoidal.ImportFromProj4(
        '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 ' + 
        '+datum=WGS84 +units=m +no_defs')

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 name: str = None,
                 cube: np.ndarray = None,
                 dayXref: dict = None,
                 logger: logging.RootLogger = None):
        
        self._cube: np.ndarray = cube
        
        # ---
        # This would be better as structured array.
        #  https://numpy.org/doc/stable/user/basics.rec.html
        # ---
        self._dayXref: dict = dayXref or {}  # {yyyyddd: index} into the cube
        self._logger: logging.RootLogger = logger
        self._name: str = name
        
    # ------------------------------------------------------------------------
    # createDs
    # ------------------------------------------------------------------------
    def createDs(self, outName: Path) -> gdal.Dataset:
        
        if type(self.cube) is np.ndarray:

            # Translate Numpy data types to GDAL types.  
            dataType = \
                gdal_array.NumericTypeCodeToGDALTypeCode(self.cube.dtype)
        
            ds = gdal.GetDriverByName('GTiff').Create(
                str(outName),
                self.cube.shape[1],
                self.cube.shape[2],
                self.cube.shape[0],
                dataType,
                options=['COMPRESS=LZW', 'BIGTIFF=YES'])

            ds.SetSpatialRef(Band.modisSinusoidal)
            ds.SetMetadata({Band.METADATA_BAND_NAME: self.name})
            return ds

        else:
            raise RuntimeError('Cannot create a data set for an empty band.')
            
    # ------------------------------------------------------------------------
    # cube
    # ------------------------------------------------------------------------
    @property
    def cube(self):
        return self._cube
        
    # ------------------------------------------------------------------------
    # dayXref
    # ------------------------------------------------------------------------
    @property
    def dayXref(self):
        return self._dayXref
        
    # ------------------------------------------------------------------------
    # getDay
    # ------------------------------------------------------------------------
    def getDay(self, day: str) -> np.ndarray:
        
        index = self.dayXref[day]
        return self.cube[index]
        
    # ------------------------------------------------------------------------
    # name
    # ------------------------------------------------------------------------
    @property
    def name(self):
        return self._name
        
    # ------------------------------------------------------------------------
    # read
    #
    # Treat everything as 16-bit integers.  NaN is a float, so use the 
    # no-data value.
    # ------------------------------------------------------------------------
    def read(self, bandFileName: Path) -> None:
        
        if self._logger:
            self._logger.info('Reading ' + str(bandFileName))
            
        gif = GeospatialImageFile(str(bandFileName))
        
        # Older band files do not have this metadata.
        self._name = gif.getDataset().GetMetadataItem(Band.METADATA_BAND_NAME)

        if not self._name:
            self._name = bandFileName.stem
            
        numDays = gif.getDataset().RasterCount
        first = True
        dayBandDt = None
        numpyType = None
        
        for dayIndex in range(numDays):
            
            dayBand = gif.getDataset().GetRasterBand(dayIndex+1)
            
            if first:
                
                first = False
                shape = (numDays, Band.ROWS, Band.COLS)
                self._cube = np.ndarray(shape, np.int16)
            
            dayNan = dayBand.ReadAsArray()
            dayNoData = np.where(np.isnan(dayNan), Band.NO_DATA, dayNan)
            day = dayNoData.astype(np.int16)
            dayKey = dayBand.GetMetadataItem(Band.METADATA_BAND_NAME)
            self._dayXref[dayKey] = dayIndex
            self._cube[dayIndex] = day

    # ------------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------------
    def write(self, outDir: Path, name: str = None, day: int = None) -> Path:

        name = name or self.name
        
        if not outDir or not outDir.exists() or not outDir.is_dir():
            raise RuntimeError('A valid output directory must be provided.')

        outName = outDir / (name + '.tif')
        
        if self._logger:
            self._logger.info('Writing ' + str(outName))
            
        if not outName.exists():
            
            ds = self.createDs(outName)
            outBandIndex = 0
        
            daysToWrite = day or self.dayXref

            for dayKey in daysToWrite:

                cubeIndex: int = self.dayXref[dayKey]
                day: np.ndarray = self.cube[cubeIndex]
                outBandIndex += 1
                gdBand = ds.GetRasterBand(outBandIndex)
                gdBand.WriteArray(day)
                gdBand.SetMetadata({Band.METADATA_BAND_NAME: dayKey})
                gdBand.SetNoDataValue(Band.NO_DATA)
                gdBand.FlushCache()
                gdBand = None

            ds = None

        return outName
