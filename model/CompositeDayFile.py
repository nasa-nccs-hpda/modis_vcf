
import logging
import os
import pathlib as Path
import sys

import numpy as np

from osgeo import gdal
from osgeo import gdal_array
from osgeo.osr import SpatialReference

from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.ProductType import ProductType


# ----------------------------------------------------------------------------
# Class CompositeDayFile
#
# This class represents a tid/band with a composite day file.  It makes 32-day
# composites.
#
# TODO: Is the xref needed?
# TODO: Make DayFile base class.
# TODO: Generalize the directory structure for all MODIS. 
# ----------------------------------------------------------------------------
class CompositeDayFile(object):

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
                 dayDir: Path = None,
                 daysInComposite: int = 32,
                 debug: bool = False,
                 logger: logging.RootLogger = None):

        self._tid: str = tid  #  Needs validation
        self._year: int = year  # Needs validation
        self._day: int = day  # Needs validation
        self._bandName: str = bandName  # Needs validation
        self._daysInComp: int = daysInComposite  # Needs validation
        self._debug: bool = debug
        
        # Logger
        if not logger:
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            if (not logger.hasHandlers()):

                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                logger.addHandler(ch)

        self._logger: logging.RootLogger = logger
        
        if not outDir.exists():
            
            raise RuntimeError('Output directory, ' + 
                               str(outDir) + 
                               ' does not exist.')
        
        self._outName: Path = outDir / \
                              (productType.productType +
                               '-' +
                               tid +
                               '-' +
                               str(self._year) +
                               str(self._day).zfill(3) +
                               '-' +
                               bandName +
                               '.bin')

        # ---
        # Day directory
        # ---
        if not dayDir:
            
            dayDir = outDir / '1-Days'
            dayDir.mkdir(parents=True, exist_ok=True)
            
        if not dayDir or not dayDir.exists() or not dayDir.is_dir():
            
            raise RuntimeError('Day directory, ' + 
                               str(dayDir) + 
                               ', does not exist.')
            
        self._dayDir: Path = dayDir

        # ---
        # The product type can change, so far, in only one case.  That happens
        # because MOD09 gets the thermal band, 31, from MOD44.
        # ---
        self._productType: ProductType = \
            productType.getProductTypeForBand(self._bandName)
            
        self._raster: np.ndarray = None
                
    # ------------------------------------------------------------------------
    # createComposite
    # ------------------------------------------------------------------------
    def _createComposite(self) -> np.ndarray:
        
        if self._outName.exists():
            
            self._logger.info('Reading composite from ' + str(self._outName))

            outBand = np.fromfile(self._outName, dtype=np.int16). \
                      reshape(ProductType.ROWS, ProductType.COLS)
                                                    
            return outBand

        self._logger.info('Computing composite for ' + 
                          self._tid + 
                          ' ' + 
                          self._bandName +
                          ' ' + 
                          str(self._year) + 
                          str(self._day).zfill(3))

        # ---
        # Determine the julian days the product type is expected to have
        # within the composite range.  
        # ---
        daysToFind = self._getDaysToFind()

        # Build the output structure.
        shp = (len(daysToFind), ProductType.ROWS, ProductType.COLS)
        dayArray = np.empty(shp)
        
        # Get each day's file, and add it to the composite.
        dayIndex = 0
        
        for year, day in daysToFind:

            bdf = BandDayFile(self._productType, 
                              self._tid, 
                              year, 
                              day, 
                              self._bandName,
                              self._dayDir,
                              debug=self._debug)

            try:
                
                # Use float because of the forthcoming mean operation.
                dayNoData = bdf.getRaster.astype(np.float64)
                          
            except RuntimeError as e:

                msg = 'Substituting empty day due to: ' + str(e)
                self._logger.warning(msg)
                
                # Substitute all no-data values for missing days.
                dayNoData = np.full((ProductType.ROWS, ProductType.COLS),
                                    ProductType.NO_DATA, 
                                    dtype=np.float64)

            dayNan = np.where(dayNoData == ProductType.NO_DATA, 
                              np.nan, 
                              dayNoData)

            dayArray[dayIndex] = dayNan
            dayIndex += 1
            
        # Compute and write the composite.
        comp = np.nanmean(dayArray, axis=0)
        
        self._raster = np.where(np.isnan(comp), 
                                ProductType.NO_DATA, 
                                comp).astype(np.int16)
        
        self._raster.tofile(self._outName)
        
        if self._debug:
            self.toTif()
            
        return self._raster
    
    # ------------------------------------------------------------------------
    # getDaysToFind
    # ------------------------------------------------------------------------
    def _getDaysToFind(self) -> list:
        
        day = self._day
        year = self._year
        numToFind = int(self._daysInComp / self._productType.dayStep)
        y1Days = []
        y2Days = []
        
        if day in self._productType.yearOneDays:
        
            dayIndex = self._productType.yearOneDays.index(day)
            numAvailable = len(self._productType.yearOneDays)
            numInYear = min(dayIndex + numToFind, numAvailable)
            y1Days = self._productType.yearOneDays[dayIndex : numInYear]
            y1Days = [(year, d) for d in y1Days]
            numMissing = numToFind - len(y1Days)
            
            # This means we are going to year two.
            if numMissing:
                
                day = self._productType.yearTwoStartDay
                numToFind = numMissing
                year = self._year + 1
                
        if day in self._productType.yearTwoDays:
        
            dayIndex = self._productType.yearTwoDays.index(day)
            numAvailable = len(self._productType.yearTwoDays)
            numInYear = min(dayIndex + numToFind, numAvailable)
            y2Days = self._productType.yearTwoDays[dayIndex : numInYear]
            y2Days = [(year, d) for d in y2Days]

        daysToFind = y1Days + y2Days
        
        return daysToFind

    # ------------------------------------------------------------------------
    # getRaster
    # ------------------------------------------------------------------------
    @property
    def getRaster(self) -> np.ndarray:
        
        if not type(self._raster) is np.ndarray:
            self._raster = self._createComposite()

        return self._raster

    # ------------------------------------------------------------------------
    # toTif
    # ------------------------------------------------------------------------
    def toTif(self) -> None:
        
        outName = self._outName.with_suffix('.tif')        
        print('Writing ' + str(outName))
        
        raster = self.getRaster

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
        gdBand = ds.GetRasterBand(1)
        gdBand.WriteArray(raster)
        gdBand.SetNoDataValue(self._productType.NO_DATA)
        gdBand.SetMetadata({'name': self._bandName})
        gdBand.FlushCache()
        gdBand = None
        ds = None
                