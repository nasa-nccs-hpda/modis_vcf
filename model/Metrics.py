
import inspect
import logging
import math
from collections import namedtuple
import numpy as np
import os
from pathlib import Path

import numpy as np
import numpy.ma as ma

from osgeo import gdal

from modis_vcf.model.Band import Band
from modis_vcf.model.CollateBandsByDate import CollateBandsByDate
from modis_vcf.model.Pair import Pair


# ----------------------------------------------------------------------------
# Class Metrics
#
# Training data: /explore/nobackup/projects/ilab/data/MODIS/MODIS_VCF/Mark_training/VCF_training_adjusted/tile_adjustment/v5.0.3samp/
# ----------------------------------------------------------------------------
class Metrics(object):

    Metric = namedtuple('Metric', 'name, desc, value')
    NDVI = 'NDVI'

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
            
        # self._outDir = outDir / (tileId + '-' + str(year))
        self._outDir = outDir / tileId

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

            self._logger.info('Output dir: ' + str(self._outDir))
            self._logger.info('Uncombined dir: ' + str(self._uncombinedDir))
            self._logger.info('Combined dir: ' + str(self._combinedDir))
            self._logger.info('Metrics dir: ' + str(self._metricsDir))

        # Other members
        self._cbbd: CollateBandsByDate = \
            CollateBandsByDate(tileId, 
                               year, 
                               inputDir, 
                               self._uncombinedDir, 
                               logger)
        
        self._periodsToCombine: int = 2
        self._availableMetrics: list = None
        self._ndvi: Band = None
        self._band31: Band = None
                
        # Combined bands
        self._bands = {}  # {bandName, Band}
        
        # ---
        # This executes @property members, unfortunately.
        # "Introspection is a sign of a bad design," said some famous 
        # software engineer.  Each metric should be a subclass.
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
    # band31
    # ------------------------------------------------------------------------
    @property
    def band31(self) -> Band:

        if not self._band31:

            bandFileName = self._combinedDir / (Pair.BAND31 + '.tif')
            
            if bandFileName.exists():
                
                self._band31 = Band()
                self._band31.read(bandFileName)
                
            else:
                
                raw31 = self.getBand(Pair.BAND31)
                self._band31.write(self._combinedDir)

        return self._band31
        
    # ------------------------------------------------------------------------
    # combine
    # ------------------------------------------------------------------------
    def _combine(self, band: Band) -> Band:
        
        if self._periodsToCombine == 1:
            return band
            
        # ---
        # This splits the 23 days to lists that are 
        # _periodsToCombine x 4800 x 4800.
        # ---
        numSplits = math.ceil(band.cube.shape[0] / self._periodsToCombine)
        
        # Keep track of days by splitting dayXref.
        sortedDays = sorted(list(band.dayXref.keys()))  # [yyyydd, ...]
        
        # [[yyyydd, yyyydd], [yyyydd, yyyydd], ...]
        splitDays = np.array_split(sortedDays, numSplits) 
        splitCubeXref = {}  # yyyydd: index into cube
        splitCubeIndex = 0
        splitCube = np.full((len(splitDays), Band.ROWS, Band.COLS), np.nan)
        
        for days in splitDays:
            
            dayArray = np.empty((self._periodsToCombine, Band.ROWS, Band.COLS))
            dayIndex = 0
            
            # Get the days of which to take the mean.
            for day in days:
                
                dayNoData = band.getDay(day).astype(np.float)
                dayNan = np.where(dayNoData == Band.NO_DATA, np.nan, dayNoData)
                dayArray[dayIndex] = dayNan
                dayIndex += 1
                
            splitCubeXref[days[0]] = splitCubeIndex
            combo = np.nanmean(dayArray, axis=0)
            splitCube[splitCubeIndex] = combo
            splitCubeIndex += 1

        splitBand = Band(band.name, splitCube, splitCubeXref, self._logger)

        return splitBand

    # ------------------------------------------------------------------------
    # getBand
    # ------------------------------------------------------------------------
    def getBand(self, bandName: str) -> Band:
        
        if bandName not in self._bands:

            bandFileName = self._combinedDir / (bandName + '.tif')
            
            if bandFileName.exists():
                
                band = Band()
                band.read(bandFileName)
                
                nanCube = np.where(band.cube == Band.NO_DATA, 
                                   np.nan, 
                                   band.cube)
                
                nanBand = Band(band.name, nanCube, band.dayXref, self._logger)
                self._bands[bandName] = nanBand
                
            else:
                
                band = self._cbbd.getBand(bandName)
                combined: Band = self._combine(band)
                self._bands[bandName] = combined
                
                noDataCube = np.where(np.isnan(combined.cube), 
                                      Band.NO_DATA, 
                                      combined.cube).astype(np.int16)
                
                noDataBand = Band(combined.name,
                                  noDataCube,    
                                  combined.dayXref, 
                                  combined._logger)
                
                noDataBand.write(self._combinedDir)
            
        return self._bands[bandName]
        
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
        
        metricFileName = self._getOutName(metricName)
        metric = None
        
        if metricFileName.exists():
            
            metric = Band()
            metric.read(metricFileName)
            
        else:
            
            mFunc = self.availableMetrics[metricName]
            metrics = mFunc()
            self.writeMetrics(metrics)
            
        return metric
        
    # ------------------------------------------------------------------------
    # ndvi
    # ------------------------------------------------------------------------
    @property
    def ndvi(self) -> Band:

        if not self._ndvi:

            ndviFileName = self._combinedDir / (Metrics.NDVI + '.tif')
            
            if ndviFileName.exists():
                
                self._ndvi = Band()
                self._ndvi.read(ndviFileName)
            
            else:
                
                if self._logger:
                    self._logger.info('Computing NDVI')

                b1 = self.getBand(Pair.BAND1)
                b2 = self.getBand(Pair.BAND2)
            
                ndviDayXref = {}  # yyyydd: index into cube
                ndviDayIndex = 0
            
                ndviCube = np.full((b1.cube.shape[0], Band.ROWS, Band.COLS),
                                   np.nan)
            
                # for day in b1Days:
                for day in b1.dayXref.keys():
            
                    b1Day = b1.cube[b1.dayXref[day]]
                    b2Day = b2.cube[b2.dayXref[day]]

                    ndviUnfiltered = \
                        (((b2Day - b1Day) / (b2Day + b1Day)) * \
                         1000).astype(np.int16)

                    ndviDay = np.where(b1Day + b2Day != 0, ndviUnfiltered, 0)
                    ndviDayXref[day] = ndviDayIndex
                    ndviCube[ndviDayIndex] = ndviDay
                    ndviDayIndex += 1

                self._ndvi = Band(Metrics.NDVI, ndviCube, ndviDayXref)
                self._ndvi.write(self._combinedDir)

        return self._ndvi

    # ------------------------------------------------------------------------
    # printAvailableMetrics
    # ------------------------------------------------------------------------
    def printAvailableMetrics(self) -> None:

        for item in self.availableMetrics().items():
            print(item[0], ':', item[1])

    # ------------------------------------------------------------------------
    # sortByReflectance
    # ------------------------------------------------------------------------
    def _sortByReflectance(self, band: Band) -> np.ndarray:
        
        ascIndexes = np.argsort(band.cube, axis=0)
        sc = np.take_along_axis(band.cube, ascIndexes, axis=0)
        return sc

    # ------------------------------------------------------------------------
    # sortByNDVI
    # ------------------------------------------------------------------------
    def _sortByNDVI(self, cube: np.ndarray) -> np.ndarray:
        
        ascIndexes = np.argsort(self.ndvi.cube, axis=0)
        sc = np.take_along_axis(cube, ascIndexes, axis=0)
        return sc

    # ------------------------------------------------------------------------
    # sortByThermal
    # ------------------------------------------------------------------------
    def _sortByThermal(self, cube: np.ndarray) -> np.ndarray:
        
        ascIndexes = np.argsort(self.band31.cube, axis=0)
        sc = np.take_along_axis(cube, ascIndexes, axis=0)
        return sc
        
    # ------------------------------------------------------------------------
    # writeMetrics
    # ------------------------------------------------------------------------
    def writeMetrics(self, metrics: list, name: str = None) -> None:

        name = name or metrics[0].name.split('-')[0]
        outName = self._metricsDir / (name + '.tif')

        ds = gdal.GetDriverByName('GTiff').Create(
            str(outName),
            Band.ROWS,
            Band.COLS,
            len(metrics),
            gdal.GDT_Float64,
            options=['COMPRESS=LZW'])

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
    def metricUnsortedMonthlyBands(self):
        
        if self._logger:

            desc = 'unsorted monthly bands (1 – 7 and NDVI) = 96 metrics'
            self._logger.info('Running ' + desc)
            
        for bandName in Pair.BANDS:
            value = self.getBand(bandName).cube
            
        self.ndvi
        
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

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            nanValue = np.nanmin(nanBand, axis=0)
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)

            name = baseName + '-' + bandName
            desc = baseName + ' ' + bandName
            metrics.append(Metrics.Metric(name, desc, value))

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

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            nanValue = np.median(nanBand, axis=0)
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + ' ' + bandName
            metrics.append(Metrics.Metric(name, desc, value))
            
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

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            nanValue = np.max(nanBand, axis=0)
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + ' ' + bandName
            metrics.append(Metrics.Metric(name, desc, value))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMaxGreenness
    # ------------------------------------------------------------------------
    def metricBandReflMaxGreenness(self):
        
        if self._logger:

            desc = 'band x reflectance associated with peak ' + \
                   'greenness = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metricsToWrite = []
        baseName = 'BandReflMaxGreenness'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            sortedCube = self._sortByNDVI(nanBand)
            nanValue = sortedCube[-1, :, :]
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + ' ' + bandName
            metricsToWrite.append(Metrics.Metric(name, desc, value))
            
        self.writeMetrics(metricsToWrite, baseName)
        
    # ------------------------------------------------------------------------
    # metricBandReflMedianGreenness
    # ------------------------------------------------------------------------
    def metricBandReflMedianGreenness(self):
        
        if self._logger:

            desc = 'band x reflectance associated with median ' + \
                   'greenness = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metricsToWrite = []
        baseName = 'BandReflMedianGreenness'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            sortedCube = self._sortByNDVI(nanBand)
            
            numBands = sortedCube.shape[0]
            midIndex = math.floor(numBands / 2)

            if midIndex % 2:
                
                # Odd number of bands
                v1 = sortedCube[midIndex-1, :, :]
                v2 = sortedCube[midIndex, :, :]
                nanValue = int(v1 / v2)
                
            else:
                
                # Even number of bands
                nanValue = sortedCube[midIndex, :, :]
            
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + ' ' + bandName
            metricsToWrite.append(Metrics.Metric(name, desc, value))
            
        self.writeMetrics(metricsToWrite, baseName)
        
    # ------------------------------------------------------------------------
    # metricBandReflMinGreenness
    # ------------------------------------------------------------------------
    def metricBandReflMinGreenness(self):
        
        if self._logger:

            desc = 'band x reflectance associated with minimum ' + \
                   'greenness = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metricsToWrite = []
        baseName = 'BandReflMinGreenness'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            sortedCube = self._sortByNDVI(nanBand)
            nanValue = sortedCube[0, :, :]
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + ' ' + bandName
            metricsToWrite.append(Metrics.Metric(name, desc, value))
            
        self.writeMetrics(metricsToWrite, baseName)
        
    # ------------------------------------------------------------------------
    # metricBandReflMaxTemp
    # ------------------------------------------------------------------------
    def metricBandReflMaxTemp(self):
        
        if self._logger:

            desc = 'band x reflectance associated with max ' + \
                   'surface temp = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metricsToWrite = []
        baseName = 'BandReflMaxTemp'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            sortedCube = self._sortByThermal(nanBand)
            nanValue = sortedCube[-1, :, :]
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + ' ' + bandName
            metricsToWrite.append(Metrics.Metric(name, desc, value))
            
        self.writeMetrics(metricsToWrite, baseName)
        
    # ------------------------------------------------------------------------
    # metricBandReflMedianTemp
    # ------------------------------------------------------------------------
    def metricBandReflMedianTemp(self):
        
        if self._logger:

            desc = 'band x reflectance associated with median ' + \
                   'surf temp = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metricsToWrite = []
        baseName = 'BandReflMedianTemp'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            sortedCube = self._sortByThermal(nanBand)
            
            numBands = sortedCube.shape[0]
            midIndex = math.floor(numBands / 2)

            if midIndex % 2:
                
                # Odd number of bands
                v1 = sortedCube[midIndex-1, :, :]
                v2 = sortedCube[midIndex, :, :]
                nanValue = int(v1 / v2)
                
            else:
                
                # Even number of bands
                nanValue = sortedCube[midIndex, :, :]
            
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + ' ' + bandName
            metricsToWrite.append(Metrics.Metric(name, desc, value))
            
        self.writeMetrics(metricsToWrite, baseName)
        
    # ------------------------------------------------------------------------
    # metricBandReflMinTemp
    # ------------------------------------------------------------------------
    def metricBandReflMinTemp(self):
        
        if self._logger:

            desc = 'band x reflectance associated with min ' + \
                   'surface temp = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metricsToWrite = []
        baseName = 'BandReflMinTemp'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            sortedCube = self._sortByThermal(nanBand)
            nanValue = sortedCube[0, :, :]
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + ' ' + bandName
            metricsToWrite.append(Metrics.Metric(name, desc, value))
            
        self.writeMetrics(metricsToWrite, baseName)
        
    # ------------------------------------------------------------------------
    # lowestMeanBandRefl
    # ------------------------------------------------------------------------
    def _lowestMeanBandRefl(self, numBands: int):
        
        if self._logger:

            desc = 'Mean of ' + \
                   str(numBands) + \
                   ' lowest band x reflectance = 7 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metricsToWrite = []
        baseName = 'Lowest' + str(numBands) + 'MeanBandRefl'
        
        for bandName in Pair.BANDS:

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            sortedBand = np.sort(nanBand, axis=0)
            slicedBand = sortedBand[0:numBands, :, :]
            nanValue = np.mean(slicedBand, axis=0)
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + bandName
            metricsToWrite.append(Metrics.Metric(name, desc, value))
            
        self.writeMetrics(metricsToWrite, baseName)
        
    # ------------------------------------------------------------------------
    # metricLowest3MeanBandRefl
    # ------------------------------------------------------------------------
    def metricLowest3MeanBandRefl(self):        
        self._lowestMeanBandRefl(3)
        
    # ------------------------------------------------------------------------
    # metricLowest6MeanBandRefl
    # ------------------------------------------------------------------------
    def metricLowest6MeanBandRefl(self):        
        self._lowestMeanBandRefl(6)
        
    # ------------------------------------------------------------------------
    # metricLowest8MeanBandRefl
    # ------------------------------------------------------------------------
    def metricLowest8MeanBandRefl(self):        
        self._lowestMeanBandRefl(8)
        
    # ------------------------------------------------------------------------
    # greenestMeanBandRefl
    # ------------------------------------------------------------------------
    def _greenestMeanBandRefl(self, numBands: int):
        
        if self._logger:

            desc = 'Mean of ' + \
                   str(numBands) + \
                   ' greenest band x reflectance = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metricsToWrite = []
        baseName = 'Greenest' + str(numBands) + 'MeanBandRefl'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            sortedCube = self._sortByNDVI(nanBand)
            startIndex = sortedCube.shape[0] - numBands
            slicedCube = sortedCube[startIndex:, :, :]
            nanValue = np.mean(slicedCube, axis=0)
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + bandName
            metricsToWrite.append(Metrics.Metric(name, desc, value))
            
        self.writeMetrics(metricsToWrite, baseName)
        
    # ------------------------------------------------------------------------
    # metricGreenest3MeanBandRefl
    # ------------------------------------------------------------------------
    def metricGreenest3MeanBandRefl(self):
        self._greenestMeanBandRefl(3)
        
    # ------------------------------------------------------------------------
    # metricGreenest6MeanBandRefl
    # ------------------------------------------------------------------------
    def metricGreenest6MeanBandRefl(self):
        self._greenestMeanBandRefl(6)
        
    # ------------------------------------------------------------------------
    # metricGreenest8MeanBandRefl
    # ------------------------------------------------------------------------
    def metricGreenest8MeanBandRefl(self):
        self._greenestMeanBandRefl(8)
        
    # ------------------------------------------------------------------------
    # _warmestMeanBandRefl
    # ------------------------------------------------------------------------
    def _warmestMeanBandRefl(self, numBands: int):
        
        if self._logger:

            desc = 'Mean of ' + \
                   str(numBands) + \
                   ' warmest band x reflectance = 8 metrics'
                   
            self._logger.info('Running ' + desc)
            
        metricsToWrite = []
        baseName = 'Warmest' + str(numBands) + 'MeanBandRefl'
        
        for bandName in Pair.BANDS + [Metrics.NDVI]:

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            sortedCube = self._sortByThermal(nanBand)
            startIndex = sortedCube.shape[0] - numBands
            slicedCube = sortedCube[startIndex:, :, :]
            nanValue = np.mean(slicedCube, axis=1)
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + bandName
            metricsToWrite.append(Metrics.Metric(name, desc, value))
            
        self.writeMetrics(metricsToWrite, baseName)
        
    # ------------------------------------------------------------------------
    # metricWarmest3MeanBandRefl
    # ------------------------------------------------------------------------
    def metricWarmest3MeanBandRefl(self):
        self._warmestMeanBandRefl(3)
        
    # ------------------------------------------------------------------------
    # metricWarmest6MeanBandRefl
    # ------------------------------------------------------------------------
    def metricWarmest6MeanBandRefl(self):
        self._warmestMeanBandRefl(6)
        
    # ------------------------------------------------------------------------
    # metricWarmest8MeanBandRefl
    # ------------------------------------------------------------------------
    def metricWarmest8MeanBandRefl(self):
        self._warmestMeanBandRefl(8)

    # ------------------------------------------------------------------------
    # metricAmpBandRefl
    # ------------------------------------------------------------------------
    def metricAmpBandRefl(self):
        
        if self._logger:

            desc = 'Sort band X low to high: ' + \
                   'amp == value 8 – value 1 = 8 metrics'
                   
            self._logger.info(desc)

        metricsToWrite = []
        baseName = 'AmpBandRefl'

        for bandName in Pair.BANDS + [Metrics.NDVI]:

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            minNanBand = np.min(nanBand, axis=0)
            maxNanBand = np.max(nanBand, axis=0)
            nanValue = maxNanBand - minNanBand
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + bandName
            metricsToWrite.append(Metrics.Metric(name, desc, value))
            
        self.writeMetrics(metricsToWrite, baseName)

        if self._logger:

            desc = 'Sort band X by NDVI high to low:' + \
                   ' amp == value 8 – value 1 = 8 metric'
                   
            self._logger.info(desc)

        metricsToWrite = []
        baseName = 'AmpGreenestBandRefl'

        for bandName in Pair.BANDS + [Metrics.NDVI]:

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            sortedCube = self._sortByNDVI(nanBand)
            minNanBand = np.min(sortedCube, axis=0)
            maxNanBand = np.max(sortedCube, axis=0)
            nanValue = maxNanBand - minNanBand
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + bandName
            metricsToWrite.append(Metrics.Metric(name, desc, value))
            
        self.writeMetrics(metricsToWrite, baseName)

        if self._logger:

            desc = 'Sort band X by temp high to low:' + \
                   ' amp == value 8 – value 1 = 8 metric'
                   
            self._logger.info(desc)

        metricsToWrite = []
        baseName = 'AmpWarmestBandRefl'

        for bandName in Pair.BANDS + [Metrics.NDVI]:

            noDataBand = self.getBand(bandName).cube
            nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
            sortedCube = self._sortByThermal(nanBand)
            minNanBand = np.min(sortedCube, axis=0)
            maxNanBand = np.max(sortedCube, axis=0)
            nanValue = maxNanBand - minNanBand
            value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
            
            name = baseName + '-' + bandName
            desc = baseName + bandName
            metricsToWrite.append(Metrics.Metric(name, desc, value))

        self.writeMetrics(metricsToWrite, baseName)

    # ------------------------------------------------------------------------
    # metricTempMeanWarmest3
    #
    # Same guts as _warmestMeanBandRefl.
    # ------------------------------------------------------------------------
    def metricTempMeanWarmest3(self):
        
        if self._logger:

            desc = 'Mean temp of warmest 3 months = 1 metric'
            self._logger.info(desc)

        baseName = 'TempMeanWarmest3'        
        noDataBand = self.getBand(Pair.BAND31).cube
        nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
        sortedCube = np.sort(nanBand, axis=0)
        startIndex = sortedCube.shape[0] - 3
        slicedCube = sortedCube[startIndex:, :, :]
        nanValue = np.mean(slicedCube, axis=1)
        value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
        metric = Metrics.Metric(baseName, desc, value)
        self.writeMetrics([metric], baseName)
        
    # ------------------------------------------------------------------------
    # metricTempMeanGreenest3
    #
    # Same guts as in _greenestMeanBandRefl.
    # ------------------------------------------------------------------------
    def metricTempMeanGreenest3(self):
        
        if self._logger:

            desc = 'Mean temp of greenest 3 months = 1 metric'
            self._logger.info(desc)

        baseName = 'TempMeanGreenest3'
        noDataBand = self.getBand(Pair.BAND31).cube
        nanBand = np.where(noDataBand==Band.NO_DATA, np.nan, noDataBand)
        sortedCube = self._sortByNDVI(nanBand)
        startIndex = sortedCube.shape[0] - 1
        slicedCube = sortedCube[startIndex:, :, :]
        nanValue = np.mean(slicedCube, axis=0)
        value = np.where(nanValue==np.nan, Band.NO_DATA, nanValue)
        metric = Metrics.Metric(baseName, desc, value)
        self.writeMetrics([metric], baseName)
        
        