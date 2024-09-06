
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

from modis_vcf.model.Band import Band
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
#
# TODO:  Is band xref needed?  Possibly only for tests.
# TODO:  Make DirStructure class.
# TODO:  Add "debug" mode, as in BDF and CDF?  Also, limit logging unless in
#        debug mode?
# TODO:  Thermal's minimum -32768 and conflict with -10001
# TODO:  Why does CDF write BAND31, instead of B_31 like the others?
# ----------------------------------------------------------------------------
class Metrics(object):

    Metric = namedtuple('Metric', 'name, desc, value')
    NDVI = 'NDVI'
    VERY_LOW_SORT_VALUE = -999999
    
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
                 logger: logging.RootLogger,
                 nanThreshold: int = 9999):

        # Output directory
        if not outDir or not outDir.exists() or not outDir.is_dir():
            raise RuntimeError('A valid output directory must be provided.')
            
        self._outDir = outDir / tileId / str(year)
        self._dayDir = self._outDir / '1-Days'
        self._compDir = self._outDir / '2-Composites'
        self._metricsDir = self._outDir / '3-Metrics'

        self._dayDir.mkdir(parents=True, exist_ok=True)
        self._compDir.mkdir(parents=True, exist_ok=True)
        self._metricsDir.mkdir(parents=True, exist_ok=True)

        self._nanThreshold: int = nanThreshold

        # Logger
        if not logger:
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            if (not logger.hasHandlers()):

                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                logger.addHandler(ch)

        self._logger: logging.RootLogger = logger

        self._logger.info('    Output dir: ' + str(self._outDir))
        self._logger.info('       Day dir: ' + str(self._dayDir))
        self._logger.info('Composites dir: ' + str(self._compDir))
        self._logger.info('   Metrics dir: ' + str(self._metricsDir))
        self._logger.info(' NaN threshold: ' + str(self._nanThreshold))

        # Other members
        self._productType: ProductType = productType
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

        self._logger.info('Applying threshold of ' + str(self._nanThreshold))
        
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
    def getBandCube(self, bandName: str) -> (np.ndarray, dict):

        # if bandName == self._productType.NDVI:
        #     return self.getNdvi()
        
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
            
            cdf = CompositeDayFile(self._productType, 
                                   self._tid, 
                                   year, 
                                   day, 
                                   bandName,
                                   self._compDir,
                                   self._dayDir)
                                   
            raster = cdf.getRaster.astype(np.float64)
            cube[cubeIndex] = raster
            key = str(year) + str(day).zfill(3)
            xref[key] = cubeIndex
            cubeIndex += 1
            
        cube = np.where(cube == ProductType.NO_DATA, np.nan, cube)
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
    def getNdvi(self) -> (np.ndarray, dict):

        self._logger.info('Computing NDVI')

        # The xref is always needed.
        b1, b1Xref = self.getBandCube(self._productType.BAND1)

        # Read it, if available.
        outName: Path = self._metricsDir / \
                        (self._productType.productType + '-NDVI' + '.bin')

        if outName.exists():
            
            self._logger.info('Reading ndvi from ' + str(outName))

            ndviCube = np.fromfile(outName).reshape(12, 
                                                    self._productType.ROWS,
                                                    self._productType.COLS)
                                                    
            return ndviCube, b1Xref

        # ---
        # Compute NDVI.
        # ---
        b2, b2Xref = self.getBandCube(self._productType.BAND2)

        # ---
        # Numpy division warnings are caused by NaNs and acceptable.
        # When b1 == NaN or b2 == NaN, the desired result is NaN.
        # ---
        with np.errstate(divide = 'ignore', invalid = 'ignore'):

            ndviCube = np.where(b1 + b2 != 0, 
                                ((b2 - b1) / (b2 + b1)) * 1000, 
                                0)

        ndviCube = self._applyThreshold(ndviCube)
            
        # Write NDVI.
        ndviCube.tofile(outName)

        # NDVI's xref is the same as band 1's.
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
    def _sortByNDVI(self, cube: np.ndarray, noDataLow = True) -> np.ndarray:
        
        # ---
        # The sort is invalid wherever NDVI is NaN, so make the cube NaN
        # where NDVI is NaN.
        # ---
        ndvi, nXref = self.getNdvi()
        # nanCube = np.where(np.isnan(ndvi), np.nan, cube)
        
        # ---
        # The metrics always consider NaN the greatest sort values.  Change
        # NDVI no-data values to a very low value, so they sort to the least
        # values.  This moves them out of the way, and prevents invalid values
        # from influencing the sort.
        # ---
        if noDataLow:
            ndvi = np.where(np.isnan(ndvi), Metrics.VERY_LOW_SORT_VALUE, ndvi)
        
        # Sort
        ascIndexes = np.argsort(ndvi, axis=0)
        sc = np.take_along_axis(cube, ascIndexes, axis=0)
        return sc

    # ------------------------------------------------------------------------
    # sortByThermal
    #
    # Has test
    # ------------------------------------------------------------------------
    def _sortByThermal(self, cube: np.ndarray, noDataLow = True) -> np.ndarray:
        
        b31, b31Xref = self.getBandCube(self._productType.BAND31)

        # ---
        # The sort is invalid wherever thermal is NaN, so make the cube NaN
        # where thermal is NaN.
        # ---
        # nanCube = np.where(np.isnan(b31), np.nan, cube)
        
        # ---
        # The metrics always consider NaN the greatest sort values.  Change
        # no-data values to a very low value, so they sort to the least
        # values.  This moves them out of the way, and prevents invalid values
        # from influencing the sort.
        # ---
        if noDataLow:
            b31 = np.where(np.isnan(b31), Metrics.VERY_LOW_SORT_VALUE, b31)
        
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
            
            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                
            else:
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

            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                
            else:
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

            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                
            else:
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

            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                
            else:
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

            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()

                # ---
                # Maximum sorts require NaN no-data values to be at the
                # beginning, so replace NaN with very low values.
                # ---
                # cube = np.where(np.isnan(cube),
                #                 Metrics.VERY_LOW_SORT_VALUE,
                #                 cube)
                #
                # sortedCube = np.sort(cube, axis=0)
                
            else:
                
                cube, xref = self.getBandCube(bandName)

            sortedCube = self._sortByNDVI(cube)    
            value = sortedCube[-1, :, :]
            
            noDataValue = np.where(np.isnan(value), 
                                   ProductType.NO_DATA, 
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

            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                
            else:
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

            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                # sortedCube = np.sort(cube, axis=0)
                
            else:

                cube, xref = self.getBandCube(bandName)

            sortedCube = self._sortByNDVI(cube, noDataLow=False)
            value = sortedCube[0, :, :]  # b/c sorted in noDataLow order
            
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

            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                
            else:
                cube, xref = self.getBandCube(bandName)

            sortedCube = self._sortByThermal(cube)
            value = sortedCube[-1, :, :]  # b/c sorted in noDataLow order
            
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

            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                
            else:
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

            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                
            else:
                cube, xref = self.getBandCube(bandName)

            sortedCube = self._sortByThermal(cube, noDataLow=False)
            value = sortedCube[0, :, :]
            
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
            slicedBand = sortedBand[:numBands, :, :]
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

            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                
            else:
                cube, xref = self.getBandCube(bandName)

            sortedCube = self._sortByNDVI(cube)
            slicedBand = sortedCube[-numBands:, :, :]
            value = np.nanmean(slicedBand, axis=0)
            
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

            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                
            else:
                cube, xref = self.getBandCube(bandName)

            sortedCube = self._sortByThermal(cube)
            slicedCube = sortedCube[-numBands:, :, :]
            value = np.nanmean(slicedCube, axis=0)
            
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
    # ------------------------------------------------------------------------
    def metricAmpBandRefl(self) -> list:
        
        if self._logger:

            desc = 'Sort band X low to high: ' + \
                   'amp == value 8 – value 1 = 8 metrics'
                   
            self._logger.info(desc)

        metrics = []
        baseName = 'AmpBandRefl'

        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                
            else:
                cube, xref = self.getBandCube(bandName)

            minBand = np.nanmin(cube, axis=0)
            maxBand = np.nanmax(cube, axis=0)

            # value = np.where(np.array_equal(minBand, maxBand, equal_nan=True),
            #                  np.nan,
            #                  maxBand - minBand)
            
            value = np.where(np.array_equal(minBand, maxBand, equal_nan=True),
                             0,
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
    # ------------------------------------------------------------------------
    def metricAmpGreenestBandRefl(self) -> list:
        
        if self._logger:

            desc = 'Sort band X by NDVI high to low:' + \
                   ' amp == value 8 – value 1 = 8 metric'
                   
            self._logger.info(desc)

        metrics = []
        baseName = 'AmpGreenestBandRefl'

        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                
            else:
                cube, xref = self.getBandCube(bandName)

            sortedCube = self._sortByNDVI(cube)

            # ---
            # Nanargmin and nanargmax do not always work because it sometimes
            # encounters all-nan slices and raises a value error.
            #
            # SortByNDVI() puts the band values related to NDVI NaN values
            # at the beginning of the sorted array.
            # ---
            ndvi, ndviXref = self.getNdvi()
            firstIndex = (np.isnan(ndvi).sum(axis=0))

            # Ensure the index is within the bounds.
            firstIndex = np.where(firstIndex >= sortedCube.shape[0],
                                  sortedCube.shape[0] - 1,
                                  firstIndex)

            minBand = \
                np.take_along_axis(sortedCube,
                                   firstIndex[None, :, :],
                                   axis=0).reshape(firstIndex.shape)

            maxBand = sortedCube[-1]
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
    # ------------------------------------------------------------------------
    def metricAmpWarmestBandRefl(self) -> list:
        
        if self._logger:

            desc = 'Sort band X by temp high to low:' + \
                   ' amp == value 8 – value 1 = 8 metric'
                   
            self._logger.info(desc)

        metrics = []
        baseName = 'AmpWarmestBandRefl'

        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            # ---
            # Nanargmin and nanargmax do not always work because it sometimes
            # encounters all-nan slices and raises a value error.
            # 
            # SortByNDVI() puts the band values related to NDVI NaN values
            # at the beginning of the sorted array.
            # ---
            if bandName == Metrics.NDVI:
                
                cube, xref = self.getNdvi()
                
            else:
                cube, xref = self.getBandCube(bandName)

            # sortedCube = self._sortByThermal(cube)
            # # minBand = sortedCube[0]
            #
            # # ---
            # # Nanargmin and nanargmax do not always work because it sometimes
            # # encounters all-nan slices and raises a value error.
            # #
            # # SortByThermal() puts the band values related to thermal NaN values
            # # at the beginning of the sorted array.
            # # ---
            # # thermal, tXref = self.getBandCube(ProductType.BAND31)
            # # numThermNan = np.isnan(thermal).sum(axis=0)
            #
            # numNoData = (sortedCube == ProductType.NO_DATA).sum(axis=0)
            #
            # # Ensure the index is within the bounds.
            # totalMonths = cube.shape[0]
            #
            # firstIndex = np.where(numNoData >= totalMonths,
            #                       totalMonths - 1,
            #                       numNoData)
            #
            # minBand = \
            #     np.take_along_axis(sortedCube,
            #                        firstIndex[None, :, :],
            #                        axis=0).reshape(firstIndex.shape)
            #
            # maxBand = sortedCube[-1]
            # value = abs(maxBand - minBand)

            # ---
            # Where the band is NaN, set thermal to NaN.  This prevents the
            # minimum and maximum thermal values from corresponding to a NaN
            # in the band, and limiting the amplitude value selection to 
            # valid values as much as possible.  This is effectively finding
            # the extreme thermal indexes where the band is not NaN.
            # ---
            thermal, tXref = self.getBandCube(ProductType.BAND31)
            thermal = np.where(np.isnan(cube), np.nan, thermal)

            # ---
            # Using np.expand_dims() and np.squeeze() are necessary because
            # our current version of Numpy, 1.21.5, does not have the
            # keepdims argument for np.argmax() or np.argmin().
            #
            # Numpy propagates NaNs in min() and max().  Meanwhile nanmin()
            # and nanmax() throw ValueError when encountering an all-NaN
            # slice.  To work around all this, set NaNs to high values before
            # finding the minimums, ...
            # ---
            thermalNansToHigh = np.where(np.isnan(thermal), np.inf, thermal)
            minLocs = np.argmin(thermalNansToHigh, axis=0)
            minLocs = np.expand_dims(minLocs, axis=0) 
            minValues = np.take_along_axis(cube, minLocs, axis=0).squeeze()
            
            # ---
            # ...then set NaNs to low values before finding maximums. 
            # ---
            thermalNansToLow = np.where(np.isnan(thermal), -np.inf, thermal)
            maxLocs = np.argmax(thermalNansToLow, axis=0)
            maxLocs = np.expand_dims(maxLocs, axis=0) 
            maxValues = np.take_along_axis(cube, maxLocs, axis=0).squeeze()
            
            value = abs(maxValues - minValues)
            
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
        