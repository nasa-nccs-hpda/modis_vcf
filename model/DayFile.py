
from abc import ABC
from abc import abstractmethod
import logging
from pathlib import Path
import sys

import numpy as np

from osgeo import gdal
from osgeo import gdal_array
from osgeo.osr import SpatialReference

from modis_vcf.model.ProductType import ProductType


# ----------------------------------------------------------------------------
# Class DayFile
#
# TODO: Validate tid with regex.
# TODO: geoTransform cannot be set when a DF is read from a Numpy array.
# ----------------------------------------------------------------------------
class DayFile(ABC):
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self):
        
        self._productType: ProductType = None
        self._bandName: str = None
        self._tid: str = None
        self._year: str = None
        self._day: int = None
        self._raster: np.ndarray = None
        self._geoTransform: tuple = None  # For toTif().
        self._logger: logging.RootLogger = None
        self._outDir: Path = None
        self._outName: Path = None
        
    # ------------------------------------------------------------------------
    # initFromParams
    # ------------------------------------------------------------------------
    def initFromParams(self, 
                       productType: ProductType,
                       bandName: str,
                       tid: str,
                       year: int,
                       day: int,
                       outDir: Path,
                       logger: logging.RootLogger = None):
        
        if not productType:
            raise ValueError('A product type must be specified.')
            
        # if not bandName or \
        #     bandName not in ProductType.BANDS + [ProductType.BAND31]:

        if not bandName or bandName not in productType.bandXref.keys():
            raise ValueError('A valid band name must be specified.')
            
        if not tid:
            raise ValueError('A tile ID must be specified.')
            
        if not year or year < 0:
            raise ValueError('A valid year must be specified.')
            
        if not day or day < 0 or day > 366:
            raise ValueError('A valid day must be specified.')
            
        self._productType = productType
        self._bandName = bandName
        self._tid = tid
        self._year = year
        self._day = day
        self._logger = logger

        # Directories
        if not outDir or not outDir.exists() or not outDir.is_dir():
            raise ValueError('A valid output directory my be specified.')
            
        self._outDir = outDir
        
        # Logger
        if not logger:

            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            if (not logger.hasHandlers()):

                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                logger.addHandler(ch)

        self._logger: logging.RootLogger = logger

        # ---
        # Returning self enables syntax like 
        # bdf = BandDayFile().initFromParams(...)
        # ---
        return self
        
    # ------------------------------------------------------------------------
    # copy
    #
    # This method returns an identical DayFile, but for a new year and day.
    # ------------------------------------------------------------------------
    def copy(self, otherDayFile, year: int, day: int):
        
        self.initFromParams(otherDayFile.productType,
                            otherDayFile.bandName,
                            otherDayFile.tid,
                            year,
                            day,
                            otherDayFile._outDir)
                            
        # Returning self enables syntax like bdf = BandDayFile().copy(...)
        return self
        
    # ------------------------------------------------------------------------
    # bandName
    # ------------------------------------------------------------------------
    @property
    def bandName(self) -> str:
        return self._bandName
        
    # ------------------------------------------------------------------------
    # day
    # ------------------------------------------------------------------------
    @property
    def day(self) -> str:
        return self._day
        
    # ------------------------------------------------------------------------
    # getRaster
    # ------------------------------------------------------------------------
    @abstractmethod
    def _getRaster(self, applyQa: bool = True) -> np.ndarray:
        pass
        
    # ------------------------------------------------------------------------
    # myName
    # ------------------------------------------------------------------------
    @abstractmethod
    def _myName(self) -> str:
        pass
        
    # ------------------------------------------------------------------------
    # outName
    # ------------------------------------------------------------------------
    @property
    def outName(self) -> Path:

        if not self._outName:
            
            self._outName: Path = self._outDir / \
                                  (self.productType.productType +
                                   '-' +
                                   self.tid +
                                   '-' +
                                   str(self.year) +
                                   str(self.day).zfill(3) +
                                   '-' +
                                   self.bandName +
                                   '.bin')
            
        return self._outName
        
    # ------------------------------------------------------------------------
    # productType
    # ------------------------------------------------------------------------
    @property
    def productType(self) -> str:
        return self._productType
        
    # ------------------------------------------------------------------------
    # raster
    # ------------------------------------------------------------------------
    def raster(self, applyQa: bool = True) -> np.ndarray:

        if not type(self._raster) is np.ndarray:

            if self.outName.exists():
            
                self._logger.info('Reading ' + \
                                  self._myName() + \
                                  ' from ' + \
                                  str(self.outName))

                outBand = np.fromfile(self._outName, dtype=np.int16). \
                          reshape(ProductType.ROWS, ProductType.COLS)
                                                    
                return outBand

            self._logger.info('Computing ' + self._myName() + ' for ' + 
                              self._tid + 
                              ' ' + 
                              self._bandName +
                              ' ' + 
                              str(self._year) + 
                              str(self._day).zfill(3))
            
            self._raster = self._getRaster(applyQa)

        return self._raster
        
    # ------------------------------------------------------------------------
    # tid
    # ------------------------------------------------------------------------
    @property
    def tid(self) -> str:
        return self._tid

    # ------------------------------------------------------------------------
    # toTif
    # ------------------------------------------------------------------------
    def toTif(self, bands: dict = None) -> None:
        
        outName = self.outName.with_suffix('.tif')        
        if outName.exists(): return
        
        self._logger.info('Writing ' + str(outName))
        
        modisSinusoidal = SpatialReference()
    
        modisSinusoidal.ImportFromProj4(
            '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 ' + 
            '+datum=WGS84 +units=m +no_defs')

        dataType = \
            gdal_array.NumericTypeCodeToGDALTypeCode(self.raster().dtype)
    
        bands = bands or {}
        bands[self.bandName] = self.raster()
        numBands = len(bands)

        ds = gdal.GetDriverByName('GTiff').Create(
            str(outName),
            self.raster().shape[0],
            self.raster().shape[1],
            numBands,
            dataType,
            options=['COMPRESS=LZW', 'BIGTIFF=YES'])

        ds.SetSpatialRef(modisSinusoidal)
        
        if self._geoTransform:
            ds.SetGeoTransform(self._geoTransform)
        
        curBandNum = 1
        
        for bandName in bands:
            
            raster = bands[bandName]
            gdBand = ds.GetRasterBand(curBandNum)
            gdBand.WriteArray(raster)
            gdBand.SetNoDataValue(self.productType.NO_DATA)
            gdBand.SetMetadata({'name': bandName})
            gdBand.FlushCache()
            gdBand = None
            curBandNum += 1
            
        ds = None
        
    # ------------------------------------------------------------------------
    # year
    # ------------------------------------------------------------------------
    @property
    def year(self) -> str:
        return self._year
        
        