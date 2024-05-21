
import inspect
import logging
import math
from collections import namedtuple
import numpy as np
import os
from pathlib import Path
import pickle
import shutil

import numpy as np
import numpy.ma as ma

from osgeo import gdal

from modis_vcf.model.Band import Band
from modis_vcf.model.CollateBandsByDate import CollateBandsByDate
from modis_vcf.model.Pair import Pair


# ----------------------------------------------------------------------------
# Class Metrics
#
# Metrics must be computed considering invalid pixels.  Numpy can ignore NaNs.
# Bands, by definition, are 16-bit unsigned integers.  This class stores
# combined bands as binary files instead of recomputing them each time.  These
# binary files include NaNs not no-data values.  Whenever this class computes
# a metric or a band as a metric(, it converts float NaN no-data values and
# 16-bit integers.
#
# Furthermore, the Parquet, at least when converted to CSV for viewing, 
# does not seem to support NaN.  Therefore, the NaNs in the metrics must be
# converted back to Band.NO_DATA.
#
# TODO:  A metric should be its own class.  They can inherit from a base, and 
# it will eliminate much of this copy and paste. Metrics can have different
# data types.  This matters because of memory limits.  Another reason for 
# separate classes is that the rigorous unit tests for each metric has a lot
# of repeated code that should be in its own methods.  While this could be
# done in MetricsTestCase, it would confuse the structure with so many 
# ancillary methods that relate only to specific test methods.
#
# TODO:  The sort methods, sortByNDVI and sortByThermal, do not consider NaN.
# Numpy sorts NaNs as the greatest value.  Is this ok?
# ----------------------------------------------------------------------------
class Metrics(object):

    Metric = namedtuple('Metric', 'name, desc, value')
    NDVI = 'NDVI'
    
    # ---
    # There must be this many valid values in a pixel stack for the metric to
    # be computed at this location.
    # ---
    NO_DATA_THRESHOLD = 3

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self,
                 tileId: str, 
                 year: int, 
                 inputDir: Path, 
                 outDir: Path,
                 logger: logging.RootLogger):

        # Input directory
        if not inputDir or not inputDir.exists() or not inputDir.is_dir():
            raise RuntimeError('A valid input directory must be provided.')
            
        # Output directory
        if not outDir or not outDir.exists() or not outDir.is_dir():
            raise RuntimeError('A valid output directory must be provided.')
            
        self._outDir = outDir / (tileId + '-' + str(year))
        # self._outDir = outDir / tileId

        # ---
        # Create the output directories.
        # ---
        self._logger: logging.RootLogger = logger
        self._uncombinedDir = self._outDir / '1-uncombined-bands'
        self._combinedDir = self._outDir / '2-combined-bands'
        self._metricsDir = self._outDir / '3-metrics'
    
        for d in [self._outDir, self._uncombinedDir, self._combinedDir,
                  self._metricsDir]:

            if not os.path.exists(d):
                os.mkdir(d)

        if self._logger:

            self._logger.info('    Output dir: ' + str(self._outDir))
            self._logger.info('Uncombined dir: ' + str(self._uncombinedDir))
            self._logger.info('  Combined dir: ' + str(self._combinedDir))
            self._logger.info('   Metrics dir: ' + str(self._metricsDir))

        # Other members
        self._cbbd: CollateBandsByDate = \
            CollateBandsByDate(tileId, 
                               year, 
                               inputDir, 
                               self._uncombinedDir, 
                               logger,
                               write=True)
        
        self._periodsToCombine: int = 2
        self._nanThreshold: int = 3
        self.availableMetrics: list = None
        self._ndvi: np.ndarray = None
                
        # Combined bands use NaNs instead of no-data values.
        self._combinedCubes = {}  # {bandName, np.ndarray}
        
        # ---
        # This executes @property members, unfortunately.
        # "Introspection is a sign of a bad design," said some famous 
        # software engineer.  (Yes, Java.)  Each metric should be a subclass.
        # ---
        members = inspect.getmembers(self, predicate=inspect.ismethod)

        # ---
        # Do not make this a property or provide an accessor, or it will
        # cause an infinite loop with inspect.getmembers, above.  This is
        # bad design.  Each metric should be a subclass.
        # ---
        self.availableMetrics = \
            {m[0]:m[1] for m in members if m[0].startswith('metric')}

    # ------------------------------------------------------------------------
    # applyThreshold
    #
    # Has test
    # ------------------------------------------------------------------------
    def _applyThreshold(self, cube: np.ndarray) -> np.ndarray:
        
        # ---
        # The cube is (~11, 4800, 4800).  Whenever the first dimension has
        # fewer valid pixels than the threshold prescribes, make all pixels
        # NaN.  This causes the metric computation to essentially ignore the
        # pixel and fill it with a no-data value.
        #
        # Min threshold is 3.  Pixel by pixel, apply threshold to stack of
        # 12.  If under threshold, then the entire stack at that location
        # becomes NaN in the combined band.
        #
        # No no-data or nans can be fed to machine learning.  Any row with 
        # a no-data must be removed.  Do I already to this when writing the
        # Parquet file?  No.
        # ---
        threshed = np.where( \
            np.count_nonzero(np.isnan(cube), axis=0) > self._nanThreshold, 
            np.nan, 
            cube)
        
        return threshed
        
    # ------------------------------------------------------------------------
    # combine
    #
    # Has test
    # ------------------------------------------------------------------------
    def _combine(self, band: Band) -> (np.ndarray, dict):
        
        if self._periodsToCombine == 1:
            return band
            
        # ---
        # This splits the 23 days to lists that are 
        # _periodsToCombine x 4800 x 4800.
        # ---
        numSplits, allDays = self._getNumSplits()
        
        # Keep track of days by splitting dayXref.
        sortedDays = sorted(allDays)  # [yyyydd, ...]
        
        # [[yyyydd, yyyydd], [yyyydd, yyyydd], ...]
        splitDays = np.array_split(sortedDays, numSplits) 
        self._logger.info('Split days: ' + str(splitDays))
        splitCube = np.full((len(splitDays), Band.ROWS, Band.COLS), np.nan)

        # ---
        # Unsorted monthly bands needs to know which day each element of the 
        # first axis represents.  Save the cross-reference table here.  It
        # can be retrieved using getDayXref(), which reads the file saved
        # here.
        # ---
        splitCubeXref = {}  # yyyydd: index into cube
        splitCubeIndex = 0
        
        for days in splitDays:
            
            dayArray = np.empty((self._periodsToCombine, Band.ROWS, Band.COLS))
            dayIndex = 0
            
            # Get the days of which to take the mean.
            for day in days:

                # Substitute all no-data values for missing days.
                try:
                    dayNoData = band.getDay(day).astype(float)

                except KeyError:

                    dayNoData = np.full((Band.ROWS, Band.COLS), 
                                        Band.NO_DATA, 
                                        dtype=np.float)
                    
                dayNan = np.where(dayNoData == Band.NO_DATA, np.nan, dayNoData)
                dayArray[dayIndex] = dayNan
                dayIndex += 1
                
            combo = np.nanmean(dayArray, axis=0)

            splitCubeXref[days[0]] = splitCubeIndex
            splitCube[splitCubeIndex] = combo
            splitCubeIndex += 1

        return splitCube, splitCubeXref

    # ------------------------------------------------------------------------
    # getBandCube
    #
    # Has test
    # ------------------------------------------------------------------------
    def getBandCube(self, 
                    bandName: str, 
                    applyThreshold: bool = True) -> np.ndarray:
        
        if bandName not in self._combinedCubes or \
            self._combinedCubes[bandName] is None:

            combName, xrefName = self.getCombinedFileNames(bandName)
            
            if combName.exists():
                
                combined = np.fromfile(combName)
                numDays = self._getNumSplits()[0]
                combined = combined.reshape(numDays, Band.ROWS, Band.COLS)
                
            elif bandName == Metrics.NDVI:
                
                combined = self.getNdvi()
                
            else:

                band = self._cbbd.getBand(bandName)
                combined, dayXref = self._combine(band)
                
                if applyThreshold:
                    combined = self._applyThreshold(combined)
                
                combined.tofile(combName)
                
                # Save the day Xref.
                with open(xrefName, 'wb') as fh:
                    pickle.dump(dayXref, fh, protocol=pickle.HIGHEST_PROTOCOL)
            
            self._combinedCubes[bandName] = combined  # Float

        return self._combinedCubes[bandName]
        
    # ------------------------------------------------------------------------
    # getBaseName
    # ------------------------------------------------------------------------
    def _getBaseName(self, metricName: str) -> str:
        
        name = metricName[6:]  # Remove 'metric'.
        return name
    
    # ------------------------------------------------------------------------
    # getCombinedFileNames
    # ------------------------------------------------------------------------
    def getCombinedFileNames(self, bandName: str) -> (Path, Path):
        
        combName = self._combinedDir / (bandName + '.bin')
        xrefName = combName.with_suffix('.xref')
        return combName, xrefName
        
    # ------------------------------------------------------------------------
    # getDayXref
    #
    # Has test
    # ------------------------------------------------------------------------
    def getDayXref(self, bandName: str) -> dict:
        
        xrefName = self.getCombinedFileNames(bandName)[1]
        xref = None
        
        with open(xrefName, 'rb') as fh:
            xref = pickle.load(fh)
            
        return xref
        
    # ------------------------------------------------------------------------
    # getMyBaseName
    # ------------------------------------------------------------------------
    def _getMyBaseName(self) -> str:
        
        funcName = inspect.currentframe().f_back.f_code.co_name
        return self._getBaseName(funcName)
    
    # ------------------------------------------------------------------------
    # getNumSplits
    # ------------------------------------------------------------------------
    def _getNumSplits(self) -> (int, list):
        
        yearOne = str(self._cbbd._year)
        daysYearOne = [yearOne + str(d).zfill(3) for d in range(65, 354, 16)]
        yearTwo = str(self._cbbd._year + 1)
        daysYearTwo = [yearTwo + str(d).zfill(3) for d in range(17, 50, 16)]
        allDays = daysYearOne + daysYearTwo
        numSplits = math.ceil(len(allDays) / self._periodsToCombine)

        return numSplits, allDays
        
    # ------------------------------------------------------------------------
    # getOutName
    # ------------------------------------------------------------------------
    def _getOutName(self, metricName: str) -> Path:
        
        name = self._getBaseName(metricName)
        outPath = self._metricsDir / (name + '.tif')
        return outPath
    
    # ------------------------------------------------------------------------
    # getMetric
    #
    # A metric, comprised of statistic performed individually on a set of
    # bands, does not really fit the Band class abstraction.  However, it 
    # works so perfectly well, that we will tolerate it for now.  Perhaps
    # simply renaming Band to something like BandedFile, would suffice.
    # ------------------------------------------------------------------------
    def getMetric(self, metricName: str) -> Band:
        
        self._logger.info('Getting ' + metricName)
        metricFileName = self._getOutName(metricName)
        
        if not metricFileName.exists():
            
            # Write then read is wasteful.
            mFunc = self.availableMetrics[metricName]
            metrics = mFunc()
            self.writeMetrics(metrics)

        metric = Band()
        metric.read(metricFileName)
            
        return metric
        
    # ------------------------------------------------------------------------
    # getNdvi
    #
    # NDVI is not added to the list of bands because it is not truly an input
    # band and some metrics do not use it.
    #
    # Do NDVI thresholding here, so it is written.  When it is read, it will
    # already have the threshold applied.
    #
    # Has test
    # ------------------------------------------------------------------------
    def getNdvi(self, applyThreshold = True) -> np.ndarray:

        if self._ndvi is None:

            ndviName, xrefName = self.getCombinedFileNames(Metrics.NDVI)
            
            if ndviName.exists():

                ndvi = np.fromfile(ndviName)
                
                self._ndvi = ndvi.reshape(self._getNumSplits()[0], 
                                          Band.ROWS, 
                                          Band.COLS)

            else:

                if self._logger:
                    self._logger.info('Computing NDVI')

                b1: np.ndarray = self.getBandCube(Pair.BAND1)
                b2: np.ndarray = self.getBandCube(Pair.BAND2)
                ndviCube = np.full((b1.shape[0], Band.ROWS, Band.COLS), np.nan)

                for day in range(b1.shape[0]):

                    b1d = b1[day]
                    b2d = b2[day]

                    ndviDay = \
                        np.where(b1d + b2d != 0,
                                 ((b2d - b1d) / (b2d + b1d)) * 1000,
                                 np.nan)
                    
                    ndviCube[day] = ndviDay
                    
                if applyThreshold:
                    ndviDay = self._applyThreshold(ndviDay)
                
                # NDVI's xref is the same as band 1.
                b1Xref = self.getCombinedFileNames(Pair.BAND1)[1]
                shutil.copy(b1Xref, xrefName)
                ndviCube.tofile(ndviName)
                self._ndvi = ndviCube
                
        return self._ndvi

    # ------------------------------------------------------------------------
    # printAvailableMetrics
    # ------------------------------------------------------------------------
    def printAvailableMetrics(self) -> None:

        for item in self.availableMetrics.items():
            print(item[0])

    # ------------------------------------------------------------------------
    # sortByReflectance
    # ------------------------------------------------------------------------
    # def _sortByReflectance(self, cube: np.ndarray) -> np.ndarray:
    #
    #     ascIndexes = np.argsort(cube, axis=0)
    #     sc = np.take_along_axis(cube, ascIndexes, axis=0)
    #     return sc

    # ------------------------------------------------------------------------
    # sortByNDVI
    #
    # Has test
    # ------------------------------------------------------------------------
    def _sortByNDVI(self, cube: np.ndarray) -> np.ndarray:
        
        ascIndexes = np.argsort(self.getNdvi(), axis=0)
        sc = np.take_along_axis(cube, ascIndexes, axis=0)
        return sc

    # ------------------------------------------------------------------------
    # sortByThermal
    #
    # Has test
    # ------------------------------------------------------------------------
    def _sortByThermal(self, cube: np.ndarray) -> np.ndarray:
        
        b31 = self.getBandCube(Pair.BAND31)
        ascIndexes = np.argsort(b31, axis=0)
        sc = np.take_along_axis(cube, ascIndexes, axis=0)
        return sc
        
    # ------------------------------------------------------------------------
    # writeMetrics
    # ------------------------------------------------------------------------
    def writeMetrics(self, metrics: list, name: str = None) -> None:

        if not metrics:
            return
            
        name = name or metrics[0].name.split('-')[0]
        outName = self._metricsDir / (name + '.tif')

        ds = gdal.GetDriverByName('GTiff').Create(
            str(outName),
            Band.ROWS,
            Band.COLS,
            len(metrics),
            gdal.GDT_Int16,
            options=['BIGTIFF=YES'])

        ds.SetSpatialRef(Band.modisSinusoidal)
        outBandIndex = 0

        for metric in metrics:
            
            outBandIndex += 1
            gdBand = ds.GetRasterBand(outBandIndex)
            gdBand.WriteArray(metric.value)
            gdBand.SetMetadataItem('Name', metric.name)
            gdBand.SetMetadataItem('Description', metric.desc)
            gdBand.SetNoDataValue(Band.NO_DATA)
            gdBand.FlushCache()
            gdBand = None

        ds = None
        self._logger.info('Wrote ' + str(outName))

    # ------------------------------------------------------------------------
    # metricUnsortedMonthlyBands
    #
    # "Unsorted monthly bands (1 – 7 and NDVI) = 96 metrics"
    # ------------------------------------------------------------------------
    def metricUnsortedMonthlyBands(self) -> list:
        
        if self._logger:

            desc = 'unsorted monthly bands (1 – 7 and NDVI) = 96 metrics'
            self._logger.info('Running ' + desc)
            
        baseName = self._getMyBaseName()
        metrics = []
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:
            
            cube = self.getBandCube(bandName)
            dayXref = self.getDayXref(bandName)
            
            noDataCube = np.where(np.isnan(cube), 
                                  Band.NO_DATA, cube).astype(int)

            # ---
            # Each day is a metric.  This makes one huge file with each band's
            # days as individual metrics.
            # ---
            for day in dayXref:
                
                name = baseName + '-' + bandName + '-Day-' + str(day)
                desc = name.replace('-', ' ')
                index = dayXref[day]
                metrics.append(Metrics.Metric(name, desc, noDataCube[index]))
                
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMin
    # ------------------------------------------------------------------------
    def metricBandReflMin(self) -> list:
        
        if self._logger:

            desc = 'minimum band X value (include NDVI) = 8 metrics'
            self._logger.info('Running ' + desc)
            
        baseName = self._getMyBaseName()
        metrics = []

        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            value = np.nanmin(cube, axis=0)
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))

        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMedian
    # ------------------------------------------------------------------------
    def metricBandReflMedian(self) -> list:
        
        if self._logger:

            desc = 'median band X value (include NDVI) = 8 metrics'
            self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = self._getMyBaseName()
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            value = np.nanmedian(cube, axis=0)
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMax
    # ------------------------------------------------------------------------
    def metricBandReflMax(self) -> list:
        
        if self._logger:

            desc = 'maximum band X value (include NDVI) = 8 metrics'
            self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = self._getMyBaseName()
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            value = np.nanmax(cube, axis=0)
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMaxGreenness
    # ------------------------------------------------------------------------
    def metricBandReflMaxGreenness(self) -> list:
        
        if self._logger:

            desc = 'band x reflectance associated with peak ' + \
                   'greenness = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'BandReflMaxGreenness'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            sortedCube = self._sortByNDVI(cube)
            value = sortedCube[-1, :, :]  # b/c sorted in ascending order
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMedianGreenness
    #
    # Has test
    # ------------------------------------------------------------------------
    def metricBandReflMedianGreenness(self) -> list:
        
        if self._logger:

            desc = 'band x reflectance associated with median ' + \
                   'greenness = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'BandReflMedianGreenness'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            sortedCube = self._sortByNDVI(cube)
            numBands = sortedCube.shape[0]

            if numBands % 2:
                
                # Odd number of bands, so use the middle index.
                midIndex = math.ceil(numBands / 2)
                value = sortedCube[midIndex, :, :]
                
            else:
                
                # Even number of bands
                midIndex = math.floor(numBands / 2)
                v1 = sortedCube[midIndex, :, :]
                v2 = sortedCube[midIndex + 1, :, :]
                diff = v1 / v2
                value = v1 + diff
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMinGreenness
    # ------------------------------------------------------------------------
    def metricBandReflMinGreenness(self) -> list:
        
        if self._logger:

            desc = 'band x reflectance associated with minimum ' + \
                   'greenness = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'BandReflMinGreenness'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            sortedCube = self._sortByNDVI(cube)
            value = sortedCube[0, :, :]  # b/c sorted in ascending order
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMaxTemp
    # ------------------------------------------------------------------------
    def metricBandReflMaxTemp(self) -> list:
        
        if self._logger:

            desc = 'band x reflectance associated with max ' + \
                   'surface temp = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'BandReflMaxTemp'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            sortedCube = self._sortByThermal(cube)
            value = sortedCube[-1, :, :]  # b/c sorted in ascending order
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

                
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMedianTemp
    # ------------------------------------------------------------------------
    def metricBandReflMedianTemp(self) -> list:
        
        if self._logger:

            desc = 'band x reflectance associated with median ' + \
                   'surf temp = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'BandReflMedianTemp'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            sortedCube = self._sortByThermal(cube)
            
            numBands = sortedCube.shape[0]
            midIndex = math.floor(numBands / 2)

            if midIndex % 2:
                
                # Odd number of bands
                v1 = sortedCube[midIndex-1, :, :]
                v2 = sortedCube[midIndex, :, :]
                value = (v1 / v2).astype(np.int)
                
            else:
                
                # Even number of bands
                value = sortedCube[midIndex, :, :]
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMinTemp
    # ------------------------------------------------------------------------
    def metricBandReflMinTemp(self) -> list:
        
        if self._logger:

            desc = 'band x reflectance associated with min ' + \
                   'surface temp = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'BandReflMinTemp'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            sortedCube = self._sortByThermal(cube)
            value = sortedCube[0, :, :]  # b/c sorted in ascending order
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # lowestMeanBandRefl
    # ------------------------------------------------------------------------
    def _lowestMeanBandRefl(self, numBands: int) -> list:
        
        if self._logger:

            desc = 'Mean of ' + \
                   str(numBands) + \
                   ' lowest band x reflectance = 7 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'Lowest' + str(numBands) + 'MeanBandRefl'
        
        for bandName in Pair.BANDS:

            cube = self.getBandCube(bandName)
            sortedBand = np.sort(cube, axis=0)
            slicedBand = sortedBand[0:numBands, :, :]
            value = np.nanmean(slicedBand, axis=0)
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricLowest3MeanBandRefl
    # ------------------------------------------------------------------------
    def metricLowest3MeanBandRefl(self) -> list:        
        return self._lowestMeanBandRefl(3)
        
    # ------------------------------------------------------------------------
    # metricLowest6MeanBandRefl
    # ------------------------------------------------------------------------
    def metricLowest6MeanBandRefl(self) -> list:        
        return self._lowestMeanBandRefl(6)
        
    # ------------------------------------------------------------------------
    # metricLowest8MeanBandRefl
    # ------------------------------------------------------------------------
    def metricLowest8MeanBandRefl(self) -> list:        
        return self._lowestMeanBandRefl(8)
        
    # ------------------------------------------------------------------------
    # greenestMeanBandRefl
    # ------------------------------------------------------------------------
    def _greenestMeanBandRefl(self, numBands: int) -> list:
        
        if self._logger:

            desc = 'Mean of ' + \
                   str(numBands) + \
                   ' greenest band x reflectance = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'Greenest' + str(numBands) + 'MeanBandRefl'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            sortedCube = self._sortByNDVI(cube)  # ascending sort
            startIndex = sortedCube.shape[0] - numBands
            slicedCube = sortedCube[startIndex:, :, :]
            value = np.nanmean(slicedCube, axis=0)
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricGreenest3MeanBandRefl
    # ------------------------------------------------------------------------
    def metricGreenest3MeanBandRefl(self) -> list:
        return self._greenestMeanBandRefl(3)
        
    # ------------------------------------------------------------------------
    # metricGreenest6MeanBandRefl
    # ------------------------------------------------------------------------
    def metricGreenest6MeanBandRefl(self) -> list:
        return self._greenestMeanBandRefl(6)
        
    # ------------------------------------------------------------------------
    # metricGreenest8MeanBandRefl
    # ------------------------------------------------------------------------
    def metricGreenest8MeanBandRefl(self) -> list:
        return self._greenestMeanBandRefl(8)
        
    # ------------------------------------------------------------------------
    # _warmestMeanBandRefl
    # ------------------------------------------------------------------------
    def _warmestMeanBandRefl(self, numBands: int) -> list:
        
        if self._logger:

            desc = 'Mean of ' + \
                   str(numBands) + \
                   ' warmest band x reflectance = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'Warmest' + str(numBands) + 'MeanBandRefl'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            sortedCube = self._sortByThermal(cube)  # ascending sort
            startIndex = sortedCube.shape[0] - numBands
            slicedCube = sortedCube[startIndex:, :, :]
            value = np.nanmean(slicedCube, axis=1)
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricWarmest3MeanBandRefl
    # ------------------------------------------------------------------------
    def metricWarmest3MeanBandRefl(self) -> list:
        return self._warmestMeanBandRefl(3)
        
    # ------------------------------------------------------------------------
    # metricWarmest6MeanBandRefl
    # ------------------------------------------------------------------------
    def metricWarmest6MeanBandRefl(self) -> list:
        return self._warmestMeanBandRefl(6)
        
    # ------------------------------------------------------------------------
    # metricWarmest8MeanBandRefl
    # ------------------------------------------------------------------------
    def metricWarmest8MeanBandRefl(self) -> list:
        return self._warmestMeanBandRefl(8)

    # ------------------------------------------------------------------------
    # metricAmpBandRefl
    # ------------------------------------------------------------------------
    def metricAmpBandRefl(self) -> list:
        
        if self._logger:

            desc = 'Sort band X low to high: ' + \
                   'amp == value 8 – value 1 = 8 metrics'
                   
            self._logger.info(desc)

        metrics = []
        baseName = 'AmpBandRefl'

        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            minBand = np.nanmin(cube, axis=0)
            maxBand = np.nanmax(cube, axis=0)

            value = np.where(np.array_equal(minBand, maxBand, equal_nan=True),
                             np.nan,
                             maxBand - minBand)
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)
                                   
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricAmpGreenestBandRefl
    #
    # Has test
    # ------------------------------------------------------------------------
    def metricAmpGreenestBandRefl(self) -> list:
        
        if self._logger:

            desc = 'Sort band X by NDVI high to low:' + \
                   ' amp == value 8 – value 1 = 8 metric'
                   
            self._logger.info(desc)

        metrics = []
        baseName = 'AmpGreenestBandRefl'

        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            sortedCube = self._sortByNDVI(cube)
            minBand = sortedCube[0]

            # ---
            # NaNs are sorted as greater than real numbers, so they appear at
            # the end of the sorted cube.  The following counts the NaNs, 
            # revealing the index of the earliest NaN, then subtracts one
            # to get the last non-NaN.
            # ---
            lastNonNanIndex = (~np.isnan(sortedCube)).sum(axis=0) - 1

            maxBand = \
                np.take_along_axis(sortedCube, 
                                   lastNonNanIndex[None, :, :],
                                   axis=0).reshape(lastNonNanIndex.shape)
            
            value = maxBand - minBand
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricAmpWarmestBandRefl
    #
    # Has test
    # ------------------------------------------------------------------------
    def metricAmpWarmestBandRefl(self) -> list:
        
        if self._logger:

            desc = 'Sort band X by temp high to low:' + \
                   ' amp == value 8 – value 1 = 8 metric'
                   
            self._logger.info(desc)

        metrics = []
        baseName = 'AmpWarmestBandRefl'

        for bandName in Pair.BANDS + [Metrics.NDVI]:

            cube = self.getBandCube(bandName)
            sortedCube = self._sortByThermal(cube)
            minBand = sortedCube[0]

            # ---
            # NaNs are sorted as greater than real numbers, so they appear at
            # the end of the sorted cube.  The following counts the NaNs, 
            # revealing the index of the earliest NaN, then subtracts one
            # to get the last non-NaN.
            # ---
            lastNonNanIndex = (~np.isnan(sortedCube)).sum(axis=0) - 1

            maxBand = \
                np.take_along_axis(sortedCube, 
                                   lastNonNanIndex[None, :, :],
                                   axis=0).reshape(lastNonNanIndex.shape)

            value = maxBand - minBand
            
            noDataValue = np.where(np.isnan(value), 
                                   Band.NO_DATA, 
                                   value).astype(int)

            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))

        return metrics
        
    # ------------------------------------------------------------------------
    # metricTempMeanWarmest3
    #
    # Has test.  
    #
    # TO DO: Deal with NaNs in sortedCube.
    # ------------------------------------------------------------------------
    def metricTempMeanWarmest3(self) -> list:
        
        if self._logger:

            desc = 'Mean temp of warmest 3 months = 1 metric'
            self._logger.info(desc)

        baseName = 'TempMeanWarmest3'        
        cube = self.getBandCube(Pair.BAND31)
        sortedCube = np.sort(cube, axis=0)
        startIndex = sortedCube.shape[0] - 3
        slicedCube = sortedCube[startIndex:, :, :]
        value = np.nanmean(slicedCube, axis=0)
        
        noDataValue = np.where(np.isnan(value), 
                               Band.NO_DATA, 
                               value).astype(int)

        metric = Metrics.Metric(baseName, desc, noDataValue)
        return [metric]
        
    # ------------------------------------------------------------------------
    # metricTempMeanGreenest3
    #
    # Test underway
    #
    # TO DO: Deal with NaNs in sortedCube.
    # ------------------------------------------------------------------------
    def metricTempMeanGreenest3(self) -> list:
        
        if self._logger:

            desc = 'Mean temp of greenest 3 months = 1 metric'
            self._logger.info(desc)

        baseName = 'TempMeanGreenest3'
        cube: np.ndarray = self.getBandCube(Pair.BAND31)
        sortedCube: np.ndarray = self._sortByNDVI(cube)

        # ---
        # NaNs are sorted as greater than real numbers, so they appear at
        # the end of the sorted cube.  The following counts the NaNs, 
        # revealing the index of the earliest NaN, then subtracts one
        # to get the last non-NaN.
        # ---
        lastNonNanIndex: np.ndarray = (~np.isnan(sortedCube)).sum(axis=0) - 1
        startIndex: np.ndarray = lastNonNanIndex - 3
        import pdb
        pdb.set_trace()
        # The following is invalid syntax, but it reflects what is needed.
        # This?  https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis
        slicedCube = sortedCube[startIndex:lastNonNanIndex, :, :]
        value = np.nanmean(slicedCube, axis=0)

        noDataValue = np.where(np.isnan(value), 
                               Band.NO_DATA, 
                               value).astype(int)

        metric = Metrics.Metric(baseName, desc, noDataValue)
        return [metric]
        