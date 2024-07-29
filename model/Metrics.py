
import inspect
import logging
from collections import namedtuple
import math
import os
from pathlib import Path
import pickle
import shutil

import numpy as np
import numpy.ma as ma

from osgeo import gdal

from modis_vcf.model.CompositeDayFile import CompositeDayFile
from modis_vcf.model.ProductType import ProductType


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
# See comment in metricAmpGreenestBandRefl() for an important note regarding
# NaNs.
#
# TODO:  A metric should be its own class.  They can inherit from a base, and 
# it will eliminate much of this copy and paste. Metrics can have different
# data types.  This matters because of memory limits.  Another reason for 
# separate classes is that the rigorous unit tests for each metric has a lot
# of repeated code that should be in its own methods.  While this could be
# done in MetricsTestCase, it would confuse the structure with so many 
# ancillary methods that relate only to specific test methods.
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
                 productType: ProductType,
                 outDir: Path,
                 logger: logging.RootLogger):

        # Output directory
        if not outDir or not outDir.exists() or not outDir.is_dir():
            raise RuntimeError('A valid output directory must be provided.')
            
        self._outDir = outDir / (tileId + '-' + str(year))

        # ---
        # Create the output directories.
        # ---
        self._logger: logging.RootLogger = logger
        self._metricsDir = self._outDir / '3-Metrics'
    
        # for d in [self._outDir, self._uncombinedDir, self._combinedDir,
        #           self._metricsDir]:
        for d in [self._outDir, self._metricsDir]:

            if not os.path.exists(d):
                os.mkdir(d)

        if self._logger:

            self._logger.info('    Output dir: ' + str(self._outDir))
            self._logger.info('   Metrics dir: ' + str(self._metricsDir))

        # Other members
        self._productType: ProductType = productType
        self._nanThreshold: int = 3
        self.availableMetrics: list = None
        self._tid: str = tileId
        self._year: int = year
                
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
    # Needs V2 test
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
    # getBandCube
    #
    # Has test
    # ------------------------------------------------------------------------
    def getBandCube(self, 
                    bandName: str, 
                    applyThreshold: bool = True) -> (np.ndarray, dict):

        if bandName == self._productType.NDVI:
            return self.getNdvi(applyThreshold)
        
        # For one band, get all the days in the year and put them into a cube.
        DAYS_SOUGHT = [(self._year,  65), (self._year,  97), (self._year, 129),
                       (self._year, 161), (self._year, 193), (self._year, 225),
                       (self._year, 257), (self._year, 289), (self._year, 321),
                       (self._year, 353), (self._year + 1,  17), 
                       (self._year + 1,  49)]
        
        cube = np.full((len(DAYS_SOUGHT), 
                       self._productType.ROWS, 
                       self._productType.COLS), np.nan)
        
        xref = {}  # yyyydd: index into cube
        cubeIndex = 0

        for year, day in DAYS_SOUGHT:
            
            cdf = CompositeDayFile(self._outDir,
                                   self._productType, 
                                   self._tid, 
                                   year, 
                                   day, 
                                   bandName)
                                   
            raster = cdf.getRaster
            cube[cubeIndex] = raster
            key = str(year) + str(day).zfill(3)
            xref[key] = cubeIndex
            cubeIndex += 1
            
        if applyThreshold:
            cube = self._applyThreshold(cube)

        return cube, xref
        
    # ------------------------------------------------------------------------
    # getBaseName
    # ------------------------------------------------------------------------
    def _getBaseName(self, metricName: str) -> str:
        
        name = metricName[6:]  # Remove 'metric'.
        return name
    
    # ------------------------------------------------------------------------
    # getMyBaseName
    # ------------------------------------------------------------------------
    def _getMyBaseName(self) -> str:
        
        funcName = inspect.currentframe().f_back.f_code.co_name
        return self._getBaseName(funcName)
    
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
    def getNdvi(self, applyThreshold = True) -> (np.ndarray, dict):

        if self._logger:
            self._logger.info('Computing NDVI')

        b1, b1Xref = self.getBandCube(self._productType.BAND1)
        b2, b2Xref = self.getBandCube(self._productType.BAND2)

        ndviCube = np.full((b1.shape[0],
                            self._productType.ROWS,
                            self._productType.COLS), np.nan)

        for day in range(b1.shape[0]):

            b1d = b1[day]
            b2d = b2[day]

            ndviDay = \
                np.where(b1d + b2d != 0,
                         ((b2d - b1d) / (b2d + b1d)) * 1000,
                         np.nan)

            ndviCube[day] = ndviDay

        if applyThreshold:
            ndviCube = self._applyThreshold(ndviCube)

        # NDVI's xref is the same as band 1.
        return ndviCube, b1Xref

    # ------------------------------------------------------------------------
    # printAvailableMetrics
    # ------------------------------------------------------------------------
    def printAvailableMetrics(self) -> None:

        for item in self.availableMetrics.items():
            print(item[0])

    # ------------------------------------------------------------------------
    # sortByNDVI
    #
    # Has test
    # ------------------------------------------------------------------------
    def _sortByNDVI(self, cube: np.ndarray) -> np.ndarray:
        
        ascIndexes = np.argsort(self.getNdvi()[0], axis=0)
        sc = np.take_along_axis(cube, ascIndexes, axis=0)
        return sc

    # ------------------------------------------------------------------------
    # sortByThermal
    #
    # Has test
    # ------------------------------------------------------------------------
    def _sortByThermal(self, cube: np.ndarray) -> np.ndarray:
        
        b31, b31Xref = self.getBandCube(self._productType.BAND31)
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
            self._productType.ROWS,
            self._productType.COLS,
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
            gdBand.SetNoDataValue(self._productType.NO_DATA)
            gdBand.FlushCache()
            gdBand = None

        ds = None
        self._logger.info('Wrote ' + str(outName))

    # ------------------------------------------------------------------------
    # metricUnsortedMonthlyBands
    #
    # "Unsorted monthly bands (1 – 7 and NDVI) = 96 metrics"
    #
    # Has test
    # ------------------------------------------------------------------------
    def metricUnsortedMonthlyBands(self) -> list:
        
        if self._logger:

            desc = 'unsorted monthly bands (1 – 7 and NDVI) = 96 metrics'
            self._logger.info('Running ' + desc)
            
        baseName = self._getMyBaseName()
        metrics = []
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:
            
            cube, dayXref = self.getBandCube(bandName)
            
            noDataCube = np.where(np.isnan(cube), 
                                  self._productType.NO_DATA, cube).astype(int)

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

        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            value = np.nanmin(cube, axis=0)
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
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
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            value = np.nanmedian(cube, axis=0)
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
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
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            value = np.nanmax(cube, axis=0)
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
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
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByNDVI(cube)
            value = sortedCube[-1, :, :]  # b/c sorted in ascending order
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
                                   value).astype(int)

            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMedianGreenness
    #
    # NaN are nuisances:
    #
    # 1.            ndvi = [3, 2, NaN, 4]
    # 2.   numNdviNotNan = 3
    # 3.            cube = [44, NaN, 66, 77]
    # 4.      sortedCube = [NaN, 44, 77, 66]
    # 5.           index = int(numNdviNotNan / 2) = 1
    # 6.          median = sortedCube[index] = 44
    #
    # V2 test successful.
    # ------------------------------------------------------------------------
    def metricBandReflMedianGreenness(self) -> list:

        if self._logger:

            desc = 'band x reflectance associated with median ' + \
                   'greenness = 8 metrics'

            self._logger.info('Running ' + desc)

        metrics = []
        baseName = 'BandReflMedianGreenness'

        # Prepare thermal information for dealing with NaNs.
        ndvi, nXref = self.getNdvi()  #1
        numNdviNotNan = (~np.isnan(ndvi)).sum(axis=0)  #2
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            print('Processing', bandName)
            cube, xref = self.getBandCube(bandName)  #3
            sortedCube = self._sortByNDVI(cube)   #4

            # This is considerably slower than Numpy functions.
            value = np.empty((self._productType.ROWS, self._productType.COLS))

            for r in range(self._productType.ROWS):

                for c in range(self._productType.COLS):

                    days = sortedCube[:, r, c]
                    ndviNotNan = numNdviNotNan[r, c]  #2
                    index = int(ndviNotNan / 2)  #6
                    value[r, c] = days[index] if index > 0 else np.nan
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
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
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByNDVI(cube)
            value = sortedCube[0, :, :]  # b/c sorted in ascending order
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
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
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByThermal(cube)
            value = sortedCube[-1, :, :]  # b/c sorted in ascending order
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
                                   value).astype(int)

                
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMedianTemp
    #
    # NaN are nuisances:
    #
    # 1.         thermal = [3, 2, NaN, 4]
    # 2.  numThermNotNan = 3
    # 3.            cube = [44, NaN, 66, 77]
    # 4.      sortedCube = [NaN, 44, 77, 66]
    # 5.           index = int(numThermalNotNan / 2) = 1
    # 6.          median = sortedCube[index] = 44
    #
    # V2 test in progress.
    # ------------------------------------------------------------------------
    def metricBandReflMedianTemp(self) -> list:
        
        if self._logger:

            desc = 'band x reflectance associated with median ' + \
                   'surf temp = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'BandReflMedianTemp'
        
        # Prepare thermal information for dealing with NaNs.
        thermal, tXref = self.getBandCube(self._productType.BAND31)  # 1
        numThermNotNan = (~np.isnan(thermal)).sum(axis=0)  #2

        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            print('Processing', bandName)
            cube, xref = self.getBandCube(bandName)  # 3
            sortedCube = self._sortByThermal(cube)   # 4

            # This is considerably slower than Numpy functions.
            value = np.empty((self._productType.ROWS, self._productType.COLS))

            for r in range(self._productType.ROWS):

                for c in range(self._productType.COLS):

                    days = sortedCube[:, r, c]
                    thermNotNan = numThermNotNan[r, c]  #2
                    index = int(thermNotNan / 2)  #6
                    value[r, c] = days[index] if index > 0 else np.nan
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
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
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByThermal(cube)
            value = sortedCube[0, :, :]  # b/c sorted in ascending order
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
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
        
        for bandName in self._productType.BANDS:

            cube, xref = self.getBandCube(bandName)
            sortedBand = np.sort(cube, axis=0)
            slicedBand = sortedBand[0:numBands, :, :]
            value = np.nanmean(slicedBand, axis=0)
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
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
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByNDVI(cube)  # ascending sort
            startIndex = sortedCube.shape[0] - numBands
            slicedCube = sortedCube[startIndex:, :, :]
            value = np.nanmean(slicedCube, axis=0)
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
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
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByThermal(cube)  # ascending sort
            startIndex = sortedCube.shape[0] - numBands
            slicedCube = sortedCube[startIndex:, :, :]
            value = np.nanmean(slicedCube, axis=1)
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
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
    #
    # Has test.
    # ------------------------------------------------------------------------
    def metricAmpBandRefl(self) -> list:
        
        if self._logger:

            desc = 'Sort band X low to high: ' + \
                   'amp == value 8 – value 1 = 8 metrics'
                   
            self._logger.info(desc)

        metrics = []
        baseName = 'AmpBandRefl'

        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            minBand = np.nanmin(cube, axis=0)
            maxBand = np.nanmax(cube, axis=0)

            value = np.where(np.array_equal(minBand, maxBand, equal_nan=True),
                             np.nan,
                             maxBand - minBand)
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
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

        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByNDVI(cube)
            minBand = sortedCube[0]

            # ---
            # NaNs are sorted as greater than real numbers, so they appear at
            # the end of the sorted cube.  The following counts the NaNs, 
            # revealing the index of the earliest NaN, then subtracts one
            # to get the last non-NaN.
            # 
            # Nanargmin and nanargmax do not always work because it sometimes
            # encounters all-nan slices and raises a value error.
            # ---
            ndvi, ndviXref = self.getNdvi()
            lastNonNanIndex = (~np.isnan(ndvi)).sum(axis=0) - 1

            maxBand = \
                np.take_along_axis(sortedCube,
                                   lastNonNanIndex[None, :, :],
                                   axis=0).reshape(lastNonNanIndex.shape)

            value = abs(maxBand - minBand)
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
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

        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, cubeXref = self.getBandCube(bandName)
            sortedCube = self._sortByThermal(cube)
            minBand = sortedCube[0]

            # ---
            # NaNs are sorted as greater than real numbers, so they appear at
            # the end of the sorted cube.  The following counts the NaNs, 
            # revealing the index of the earliest NaN, then subtracts one
            # to get the last non-NaN.
            # ---
            b31, b31Xref = self.getBandCube(self._productType.BAND31)
            lastNonNanIndex = (~np.isnan(b31)).sum(axis=0) - 1

            maxBand = \
                np.take_along_axis(sortedCube, 
                                   lastNonNanIndex[None, :, :],
                                   axis=0).reshape(lastNonNanIndex.shape)

            value = abs(maxBand - minBand)
            
            noDataValue = np.where(np.isnan(value), 
                                   self._productType.NO_DATA, 
                                   value).astype(int)

            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, noDataValue))

        return metrics
        
    # ------------------------------------------------------------------------
    # metricTempMeanWarmest3
    #
    # Has test.  
    # ------------------------------------------------------------------------
    def metricTempMeanWarmest3(self) -> list:
        
        if self._logger:

            desc = 'Mean temp of warmest 3 months = 1 metric'
            self._logger.info(desc)

        baseName = 'TempMeanWarmest3'        
        thermal, tXref = self.getBandCube(self._productType.BAND31)
        
        # ---
        # Change NaN to low values so they are at the beginning of the sorted
        # array and are disregarded when selecting the warmest.
        # ---
        thermal = np.where(np.isnan(thermal), 
                           self._productType.NO_DATA, 
                           thermal)
                           
        sortedCube = np.sort(thermal, axis=0)
        
        # ---
        # There remains the cases where there are fewer than three valid 
        # values.  Create a masked array, masking the low values, take the
        # last three elements from the array, which may contain low values,
        # and compute the mean with masked_array.mean().
        # ---
        mArray = ma.masked_equal(sortedCube, self._productType.NO_DATA)
        slicedArray = mArray[-3:, :, :]
        value = slicedArray.mean(axis=0).astype(int)
        noDataValue = value.filled(self._productType.NO_DATA)
        metric = Metrics.Metric(baseName, desc, noDataValue)
        return [metric]
        
    # ------------------------------------------------------------------------
    # metricTempMeanGreenest3
    #
    # Has test.  
    # ------------------------------------------------------------------------
    def metricTempMeanGreenest3(self) -> list:
        
        if self._logger:

            desc = 'Mean temp of greenest 3 months = 1 metric'
            self._logger.info(desc)

        baseName = 'TempMeanGreenest3'
        ndvi, ndviXref = self.getNdvi()
        
        # ---
        # Change NaN to low values so they are at the beginning of the sorted
        # array and are disregarded when selecting the warmest.
        # ---
        ndvi = np.where(np.isnan(ndvi), self._productType.NO_DATA, ndvi)
        sortedCube = np.sort(ndvi, axis=0)
        
        # ---
        # There remains the cases where there are fewer than three valid 
        # values.  Create a masked array, masking the low values, take the
        # last three elements from the array, which may contain low values,
        # and compute the mean with masked_array.mean().
        # ---
        mArray = ma.masked_equal(sortedCube, self._productType.NO_DATA)
        slicedArray = mArray[-3:, :, :]
        value = slicedArray.mean(axis=0).astype(int)
        noDataValue = value.filled(self._productType.NO_DATA)
        metric = Metrics.Metric(baseName, desc, noDataValue)
        return [metric]
        