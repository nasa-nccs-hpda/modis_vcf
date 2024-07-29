
import logging
import os
import pathlib as Path
import sys

import numpy as np

from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.ProductType import ProductType


# ----------------------------------------------------------------------------
# Class CompositeDayFile
#
# This class represents a tid/band with a composite day file.  It makes 32-day
# composites.
#
# TODO:  See Metrics._combine and Metrics._applyThreshold.
# TODO:  Is the xref needed?
# ----------------------------------------------------------------------------
class CompositeDayFile(object):

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 outDir: Path,
                 productType: ProductType,
                 tileId: str,
                 year: int,
                 day: int,
                 bandName: str,
                 daysInComposite: int = 32,
                 logger: logging.RootLogger = None):

        self._tileId: str = tileId  #  Needs validation
        self._year: int = year  # Needs validation
        self._day: int = day  # Needs validation
        self._bandName: str = bandName  # Needs validation
        self._daysInComp: int = daysInComposite  # Needs validation
        self._baseDir: Path = outDir # Needs validation
        
        # Logger
        if not logger:
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            logger.addHandler(ch)

        self._logger: logging.RootLogger = logger
        
        # ---
        # Output file name.  Note which product type is used, and the
        # comment below regarding band 31.
        # ---
        outDir = outDir / '2-Composites'
        
        if not outDir.exists():
            os.mkdir(outDir)
        
        self._outName: Path = outDir / \
                              (productType.productType + 
                               '-' +
                               str(self._year) + 
                               str(self._day).zfill(3) + 
                               '-' +
                               bandName +
                               '.bin')

        # ---
        # The product type can change, so far, in only one case.  That happens
        # because MOD09 gets the thermal band, 31, from MOD44.
        # ---
        self._productType: ProductType = \
            productType.getProductTypeForBand(self._bandName)
            
        # Existing raster
        self._raster: np.ndarray = None if not self._outName.exists() else \
            np.fromfile(self._outName).reshape(ProductType.ROWS,
                                               ProductType.COLS)
                                               
    # ------------------------------------------------------------------------
    # createComposite
    # ------------------------------------------------------------------------
    def _createComposite(self) -> np.ndarray:
        
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
                              self._tileId, 
                              year, 
                              day, 
                              self._bandName,
                              self._baseDir)

            try:
                
                # Use float because of the forthcoming mean operation.
                dayNoData = bdf.getRaster.astype(float)
                          
            except RuntimeError as e:

                msg = 'Substituting empty day due to: ' + str(e)
                self._logger.warn(msg)
                
                # Substitute all no-data values for missing days.
                dayNoData = np.full((ProductType.ROWS, ProductType.COLS),
                                    ProductType.NO_DATA, 
                                    dtype=np.float)

            dayNan = np.where(dayNoData == ProductType.NO_DATA, 
                              np.nan, 
                              dayNoData)

            dayArray[dayIndex] = dayNan
            dayIndex += 1
            
        # Compute and write the composite.
        comp = np.nanmean(dayArray, axis=0)
        self._raster = comp
        comp.tofile(self._outName)
        
        return comp
    
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
        
        if type(self._raster) is np.ndarray:
            
            return self._raster
            
        else:
            return self._createComposite()
        