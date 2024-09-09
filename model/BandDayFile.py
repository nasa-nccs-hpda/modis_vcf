
import logging
import os
from pathlib import Path
import sys

import numpy as np

from osgeo import gdal
from osgeo import gdal_array
from osgeo.osr import SpatialReference

from modis_vcf.model.ProductType import ProductType


# ----------------------------------------------------------------------------
# Class BandDayFile
#
# This class represents a single day/tid/band file with the QA applied.
#
# This design does a lot of redundant reading; however, its simplicity makes
# it worthwhile.
#
# TODO:  __init__ validation
#
# TODO:  Move day from __init__ to getRaster(), then one BDF instance can be
#        reused.
#
# TODO:  After this point, ProductType is no longer needed because combining
#        can be performed based on the file name.  True?
#
# TODO:  Make _geoTransform a property
# ----------------------------------------------------------------------------
class BandDayFile(object):

    DEFAULT_ZENITH_CUTOFF = 72

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 productType: ProductType,
                 tid: str,
                 year: int,
                 day: int,
                 bandName: str,
                 outDir: Path,
                 debug: bool = False,
                 logger: logging.RootLogger = None):
        
        self._productType: ProductType = productType
        self._tid: str = tid  #  Needs validation
        self._year: int = year  # Needs validation
        self._day: int = day  # Needs validation
        self._bandName: str = bandName  # Needs validation
        self._debug: bool = debug
        self._geoTransform: tuple = None  # For toTif().
        
        if not outDir.exists():
            
            raise RuntimeError('Output directory, ' + 
                               str(outDir) + 
                               ' does not exist.')
                               
        self._outDir: Path = outDir
        
        self._outName: Path = outDir / \
                              (self._productType.productType +
                               '-' +
                               tid +
                               '-' +
                               str(self._year) +
                               str(self._day).zfill(3) +
                               '-' +
                               self._bandName +
                               '.bin')
                               
        # Logger
        if not logger:
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            if (not logger.hasHandlers()):

                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                logger.addHandler(ch)

        self._logger: logging.RootLogger = logger
        
        self._solarZenith: np.ndarray = None
        self._qaMask: np.ndarray = None
        self._raster: np.ndarray = None

    # ------------------------------------------------------------------------
    # getRaster
    # ------------------------------------------------------------------------
    @property
    def getRaster(self) -> np.ndarray:

        if not type(self._raster) is np.ndarray:
            self._raster = self._readRaster()

        return self._raster
        
    # ------------------------------------------------------------------------
    # readSubdataset
    # ------------------------------------------------------------------------
    def _readSubdataset(self, 
                        inBandName: str = None, 
                        applyNoData: bool = True) -> (np.ndarray, int):
        
        bandName = inBandName or self._bandName
        
        fileName: Path = self._productType.findFile(self._tid,
                                                    self._year,
                                                    self._day,
                                                    bandName)
        
        ds: gdal.Dataset = gdal.Open(str(fileName))
        subdatasetIndex: int = self._productType.bandXref[bandName]
        bandDs = gdal.Open(ds.GetSubDatasets()[subdatasetIndex][0])
        bandNoData = bandDs.GetRasterBand(1).GetNoDataValue()
        bandDataType = bandDs.GetRasterBand(1).DataType
        self._geoTransform: tuple = bandDs.GetGeoTransform()
        
        # ReadAsArray automatically resamples when necessary.
        rawBand = bandDs.ReadAsArray(buf_xsize=ProductType.COLS,
                                     buf_ysize=ProductType.ROWS)
        
        if applyNoData:
            
            rawBand = \
                np.where(rawBand == bandNoData, ProductType.NO_DATA, rawBand)

        return (rawBand, bandDataType)

    # ------------------------------------------------------------------------
    # readRaster
    # ------------------------------------------------------------------------
    def _readRaster(self, applyQa: bool = True) -> np.ndarray:
        
        if self._outName.exists():
            
            self._logger.info('Reading band from file.')

            outBand = np.fromfile(self._outName, dtype=np.int16). \
                      reshape(ProductType.ROWS, ProductType.COLS)
                                                    
            return outBand

        self._logger.info('Reading band from HDF for ' +
                          self._tid + 
                          ' ' + 
                          self._bandName +
                          ' ' + 
                          str(self._year) + 
                          str(self._day).zfill(3))

        # Read the raster without QA.
        outBand, dataType = self._readSubdataset()  # Int16
        
        # ---
        # If a pixel value is larger than the solar zenith cut off, clamp it
        # to 16000. Outband.dtype = 'int16'.
        # ---
        zenithCutOff: int = BandDayFile.DEFAULT_ZENITH_CUTOFF
        solz, dType = self._readSubdataset(ProductType.SOLZ)
        
        solz = (solz * self._productType.solarZenithScaleFactor). \
               astype(np.int16)
               
        # Apply the QA.  It does not use ProductType.NO_DATA.
        if applyQa:
            
            state, dtype = self._readSubdataset(ProductType.STATE, False)
        
            self._qaMask: np.ndarray = \
                self._productType.createQaMask(state, solz, zenithCutOff)

            outBand = np.where(self._qaMask==1, outBand, ProductType.NO_DATA)
            
        outBand = outBand.astype(np.int16)
        outBand.tofile(self._outName)
    
        if self._debug: 
            
            solzName = self._outName.with_suffix('.solz.tif')
            self._writeTif(solz, solzName)
            self.toTif()
            
        return outBand

    # ------------------------------------------------------------------------
    # toTif
    # ------------------------------------------------------------------------
    def toTif(self) -> None:
        
        outName = self._outName.with_suffix('.tif')        
        if outName.exists(): return
        self._writeTif(self.getRaster, outName)

    # ------------------------------------------------------------------------
    # _writeTif
    # ------------------------------------------------------------------------
    def _writeTif(self, raster: np.ndarray, outName: Path) -> None:
        
        self._logger.info('Writing ' + str(outName))
        
        modisSinusoidal = SpatialReference()
    
        modisSinusoidal.ImportFromProj4(
            '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 ' + 
            '+datum=WGS84 +units=m +no_defs')

        dataType = \
            gdal_array.NumericTypeCodeToGDALTypeCode(raster.dtype)
    
        ds = gdal.GetDriverByName('GTiff').Create(
            str(outName),
            raster.shape[0],
            raster.shape[1],
            1,
            dataType,
            options=['COMPRESS=LZW', 'BIGTIFF=YES'])

        ds.SetSpatialRef(modisSinusoidal)
        ds.SetGeoTransform(self._geoTransform)
        gdBand = ds.GetRasterBand(1)
        gdBand.WriteArray(raster)
        gdBand.SetNoDataValue(self._productType.NO_DATA)
        gdBand.SetMetadata({'name': self._bandName})
        gdBand.FlushCache()
        # gdBand = None
        # gdBand = ds.GetRasterBand(2)
        # gdBand.WriteArray(self._qaMask)
        # gdBand.SetNoDataValue(self._productType.NO_DATA)
        # gdBand.SetMetadata({'name': 'QA'})
        # gdBand.FlushCache()
        gdBand = None

        ds = None
        