
import logging
from pathlib import Path

import numpy as np

from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.DayFile import DayFile
from modis_vcf.model.ProductType import ProductType


# ----------------------------------------------------------------------------
# Class CompositeDayFile
#
# TODO: Validate input.
# ----------------------------------------------------------------------------
class CompositeDayFile(DayFile):

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self):
        
        super(CompositeDayFile, self).__init__()
        self._dayDir: Path = None
        self._daysInComp: int = 32

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
                       logger: logging.RootLogger,
                       dayDir: Path,
                       numDaysInComp: int = None):

        super().initFromParams(productType,
                               bandName,
                               tid,
                               year,
                               day,
                               outDir,
                               logger)
        
        # Day directory
        if not dayDir or not dayDir.exists() or not dayDir.is_dir():

            raise RuntimeError('Day directory, ' +
                               str(dayDir) +
                               ', does not exist.')

        self._dayDir: Path = dayDir
        
        if numDaysInComp:
            self._daysInComp: int = numDaysInComp

        return self

    # ------------------------------------------------------------------------
    # copy
    # ------------------------------------------------------------------------
    def copy(self, otherCDF, year: int, day: int):
        
        self.initFromParams(otherCDF.productType,
                            otherCDF.bandName,
                            otherCDF.tid,
                            year,
                            day,
                            otherCDF._outDir,
                            otherCDF._logger,
                            otherCDF._dayDir,
                            otherCDF._daysInComp)
                            
        return self

    # ------------------------------------------------------------------------
    # getRaster
    # ------------------------------------------------------------------------
    def _getRaster(self) -> np.ndarray:
        
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

            bdf = BandDayFile().initFromParams(self.productType,
                                               self.bandName,
                                               self.tid,
                                               year,
                                               day,
                                               self._dayDir,
                                               self._logger)
            
            try:
                
                # Use float because of the forthcoming mean operation.
                dayNoData = bdf.raster.astype(np.float64)
                          
                if not self._geoTransform: 
                    self._geoTransform = bdf._geoTransform
            
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
        
        self._raster.tofile(self.outName)
        
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
    # myName
    # ------------------------------------------------------------------------
    def _myName(self) -> str:
        return 'Composite'
        