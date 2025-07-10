
import logging
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pandas as pd

from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.DayFile import DayFile
from modis_vcf.model.ProductType import ProductType


# ----------------------------------------------------------------------------
# Class CompositeDayFile
#
# TODO: Validate input.
# ----------------------------------------------------------------------------
class CompositeDayFile(DayFile):

    SUBSTITUTE_PIXEL_VALUE = 10000

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self):

        super(CompositeDayFile, self).__init__()
        self._dayDir: Path = None
        self._daysInComp: int = 48  # 3 x 16-day BandDayFiles

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

        self.outStateName: Path = \
            self.outName.parent / \
            (self.outName.stem + '-state' + self.outName.suffix)

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
    # getCloudMaskName
    # ------------------------------------------------------------------------
    def _getCloudMaskName(self) -> Path:

        n = self.outName.parent / Path(self.outName.stem + '-cloud.npy')
        return n

    # ------------------------------------------------------------------------
    # getCloudMask
    # ------------------------------------------------------------------------
    def getCloudMask(self) -> np.ndarray:

        # Must get the raster to generate the mask.
        self._getRaster()
        mask = np.load(self._getCloudMaskName())
        return mask

    # ------------------------------------------------------------------------
    # getQaMaskName
    # ------------------------------------------------------------------------
    def _getQaMaskName(self) -> Path:

        n = self.outName.parent / Path(self.outName.stem + '-qa.npy')
        return n

    # ------------------------------------------------------------------------
    # getQaMask
    # ------------------------------------------------------------------------
    def getQaMask(self) -> np.ndarray:

        # Must get the raster to generate the mask.
        self._getRaster()
        mask = np.load(self._getQaMaskName())
        return mask

    # ------------------------------------------------------------------------
    # getRaster
    # ------------------------------------------------------------------------
    def _getRaster(self, applyQa=True) -> np.ndarray:

        # ---
        # Determine the julian days the product type is expected to have
        # within the composite range.
        # ---
        daysToFind = self._getDaysToFind()

        # ---
        # Store each BandDayFile, and make decisions about how to compute
        # their composite value later.  PGE61 uses two masks.
        # ---
        rastersWithQa = []

        for year, day in daysToFind:

            raster: np.ndarray = None
            qa: np.ndarray = None
            cloud: np.ndarray = None

            bdf = BandDayFile().initFromParams(self.productType,
                                               self.bandName,
                                               self.tid,
                                               year,
                                               day,
                                               self._dayDir,
                                               self._logger)

            try:

                # ---
                # This is amateur hour.  Band 31 should have its own BDF
                # derivation, and there should be a day file factory.
                # ---
                qa = False if self.bandName == ProductType.BAND31 else True

                raster = bdf.raster(applyQa=qa)
                state = bdf.getState()  # uint16
                solz = bdf.getSolz()

                # Like pge61's qa and cloud masks, 0 is bad and 1 is good.
                qa = self._productType.createQaMask(state,
                                                    solz,
                                                    BandDayFile.ZENITH_CUTOFF)

                cloud = self._productType.createCloudMask( \
                    state,
                    solz,
                    BandDayFile.ZENITH_CUTOFF)

                if not self._geoTransform:
                    self._geoTransform = bdf._geoTransform

            except RuntimeError as e:

                msg = 'Substituting empty day due to: ' + str(e)
                self._logger.warning(msg)

                # Substitute values for missing days.
                raster = np.full((ProductType.ROWS, ProductType.COLS),
                                 CompositeDayFile.SUBSTITUTE_PIXEL_VALUE,
                                 dtype=np.int16)

                # ---
                # In PGE61, when a raster has a substitute pixel value,
                # 10000, the two qa arrays get zeros.
                # ---
                qa = np.zeros_like(raster)
                cloud = np.zeros_like(raster)

            rastersWithQa.append((raster, qa, cloud))

        comp, qa, cloud = self._computeComposite(rastersWithQa)

        np.save(self._getQaMaskName(), qa)
        np.save(self._getCloudMaskName(), cloud)

        return comp

    # ------------------------------------------------------------------------
    # computeComposite
    # ------------------------------------------------------------------------
    def _computeComposite(self, rastersWithQa: list) -> list:

        # ---
        # Case 1:  There is at least one day that satisfies the QA. Make a
        # masked array subject to QA, then take the mean of the masked days.
        # Output QA = 0.
        #
        # Case 2:  Do the average regardless of the mask.
        # Output QA = 1.
        #
        # Case 3:  Test and mark composites with all cloudy days.
        # Output QA = 1.
        #
        # When MaskedArray's mask is True, the value is masked.
        # ---
        rows = rastersWithQa[0][0].shape[0]
        cols = rastersWithQa[0][0].shape[1]

        temp = np.full((len(rastersWithQa), rows, cols),
                       CompositeDayFile.SUBSTITUTE_PIXEL_VALUE)

        goodDayCube: ma.MaskedArray = ma.asarray(temp)
        allDayCube: np.ndarray = temp
        cloudTotal = np.zeros((rows, cols), dtype=np.int16)
        index = 0

        for raster, qa, cloud in rastersWithQa:

            # Case 1: Mask bad qa points, leave only good qa raster values.
            goodQaRaster: ma.MaskedArray = ma.masked_where(qa == 0, raster)
            goodDayCube[index] = goodQaRaster

            # ---
            # Case 2: Take the mean regardless of the QA, so include all band
            # days here.
            # ---
            allDayCube[index] = raster

            # Case 3: Accummulate cloud masks to flag all-cloudy composites.
            cloudTotal += cloud

            index += 1

        # ---
        # Case 1
        #
        # GoodDayCube has unmasked band days where qa == 1.  Taking the mean
        # of GoodDayCube will return the mean of the unmasked band days; when
        # all band days are bad, the mean will be masked.  The output composite
        # will include goodComp's unmasked values, and the masked values will
        # be filled by case 2.
        # ---
        goodComp = goodDayCube.mean(axis=0)

        # ---
        # Case 2
        #
        # Take the mean of all days.
        # ---
        allComp = allDayCube.mean(axis=0)

        # Combine goodComp and allComp based on the input qa.
        comp = np.where(goodComp.mask, allComp, goodComp).astype(np.int16)

        # The output QA is 0 for case 1 and 1 for case 2.
        outQa = np.where(goodComp.mask, 1, 0).astype(np.uint8)

        # ---
        # Case 3
        #
        # This uses the same computation as case 2, so there is no need to
        # compute anything.  The only thing that happens is that the cloud
        # mask is created.  For the input cloud mask, 0 is bad and 1 is good.
        # For the output cloud mask, 0 is good and 1 is bad.
        # ---
        outCloud = np.where(cloudTotal == 3, 1, 0).astype(np.uint8)

        return comp, outQa, outCloud

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

