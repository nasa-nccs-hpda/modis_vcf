
from concurrent.futures import ProcessPoolExecutor
import inspect
import logging
import multiprocessing
from collections import namedtuple
import math
import os
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import numpy.ma as ma

from osgeo import gdal
from osgeo.osr import SpatialReference

from core.model.GeospatialImageFile import GeospatialImageFile
from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.CompositeDayFile import CompositeDayFile
from modis_vcf.model import LowSun
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
# TODO:  Move call to getNdvi() into getBandCube().
# ----------------------------------------------------------------------------
class Metrics(object):

    # ---
    # This holds a metric's information, in lieu of a proper class.  Metric
    # computations put their results in this.  Metrics are written to file
    # from this.  Metrics are read from file into this.
    # ---
    Metric = namedtuple('Metric', 'name, desc, value')
    
    NDVI = 'NDVI'
    VERY_LOW_SORT_VALUE = -999999

    # ---
    # Greenland tiles where NDVI is set to 0 in any composite period with no
    # day under the zenith cutoff (0 < solz < 65).  Polar-night periods
    # otherwise reach the models unfiltered and give spurious tree cover.
    # Iceland (h17v02) is excluded: the fix did not lower its MAE against C6.
    # See Greenland_Iceland_Scoped_Fix_v2_Results.md.
    # ---
    ZENITH_ONLY_NDVI_TILES = ('h15v02', 'h16v00', 'h16v01', 'h16v02',
                              'h17v00', 'h17v01')
    
    # ---
    # There must be this many valid values in a pixel stack for the metric to
    # be computed at this location.
    # ---
    # NO_DATA_THRESHOLD = 3     OBSOLETE?

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self,
                 tileId: str, 
                 year: int, 
                 productType: ProductType,
                 outDir: Path,
                 logger: logging.RootLogger,
                 nanThreshold: int = 9999,
                 lowSun: str = None):

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
        self._readCache: dict = {}   # metric name -> list of Metric read from disk, so each file is read once
        self._year: int = year

        # Two sixteen-day composites
        # self._daysSought = \
        #     [(self._year,  65), (self._year,  97), (self._year, 129),
        #      (self._year, 161), (self._year, 193), (self._year, 225),
        #      (self._year, 257), (self._year, 289), (self._year, 321),
        #      (self._year, 353), (self._year + 1,  17),
        #      (self._year + 1,  49)]

        # Three sixteen-day composites
        self._daysSought = \
            [(self._year,  65), (self._year, 113), (self._year, 161),
             (self._year, 209), (self._year, 257), (self._year, 305),
             (self._year, 353), (self._year + 1,  33)]

        # ---
        # This executes @property members, unfortunately.
        # "The need for introspection is a sign of a bad design," said some
        # famous software engineer.  (Yes, Java.)  Each metric should be a
        # subclass.
        # ---
        members = inspect.getmembers(self, predicate=inspect.ismethod)

        # ---
        # Do not make this a property or provide an accessor, or it will
        # cause an infinite loop with inspect.getmembers, above.  This is
        # bad design.  Each metric should be a subclass.
        # ---
        self.availableMetrics = \
            {m[0]:m[1] for m in members if m[0].startswith('metric')}

        # ---
        # Low-sun mode: stamp this root, check its cached dailies against the
        # mode in force, and write the per-day substitution counts (LowSun).
        # ---
        # lowSun: the mode this root must have been built with, when it is not
        # the one in force (the blend partner's V4r root is unsubstituted).
        LowSun.guard(self, lowSun)
            
    # ------------------------------------------------------------------------
    # applyThreshold
    # ------------------------------------------------------------------------
    def _applyThreshold(self, cube: np.ndarray) -> ma.MaskedArray:

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
        # threshed = np.where( \
        #     np.count_nonzero(np.isnan(cube), axis=0) > self._nanThreshold,
        #     np.nan,
        #     cube)

        threshed = ma.where( \
            np.count_nonzero(np.isnan(cube), axis=0) > self._nanThreshold,
            np.nan,
            cube)
        
        return threshed
        
    # ------------------------------------------------------------------------
    # buildCompositesInParallel
    #
    # Reading HDF4 is serialized inside one process (GDAL holds a global
    # lock around the HDF4 library), so the composites of a band are built in
    # separate processes instead.  Each worker runs the same CompositeDayFile
    # code and writes the same files (2-Composites and 1-Days); getBandCube
    # then reads them as usual.  Composite periods share no input days, so
    # workers never write the same file.  Off when compositeWorkers is 1.
    # ------------------------------------------------------------------------
    compositeWorkers: int = 1

    def _buildCompositesInParallel(self, bandName: str) -> None:

        if Metrics.compositeWorkers <= 1:
            return

        todo = []

        for year, day in self._daysSought:

            cdf = CompositeDayFile().initFromParams(self._productType,
                                                    bandName,
                                                    self._tid,
                                                    year,
                                                    day,
                                                    self._compDir,
                                                    logger=self._logger,
                                                    dayDir=self._dayDir)

            if not cdf.outName.exists():
                todo.append((year, day))

        if len(todo) < 2:
            return

        self._logger.info('Building ' + str(len(todo)) + ' ' + bandName +
                          ' composites in parallel')

        workers = min(Metrics.compositeWorkers, len(todo))
        ctx = multiprocessing.get_context('spawn')

        with ProcessPoolExecutor(workers,
                                 mp_context=ctx,
                                 initializer=_initCompositeWorker,
                                 initargs=(self._logger.getEffectiveLevel(),)) \
                as executor:

            futures = [executor.submit(_buildComposite,
                                       self._productType,
                                       bandName,
                                       self._tid,
                                       year,
                                       day,
                                       self._compDir,
                                       self._dayDir)
                       for year, day in todo]

            for future in futures:
                future.result()

    # ------------------------------------------------------------------------
    # getBandCube
    #
    # Has test
    # ------------------------------------------------------------------------
    def getBandCube(self, bandName: str) -> (np.ndarray, dict):

        if bandName == Metrics.NDVI:
            return self.getNdvi()

        # For one band, get all the days in the year and put them into a cube.
        cube: np.ndarray = np.full((len(self._daysSought), 
                                   self._productType.ROWS, 
                                   self._productType.COLS), np.nan)
        
        self._buildCompositesInParallel(bandName)

        xref = {}  # yyyydd: index into cube
        cubeIndex = 0

        for year, day in self._daysSought:
            
            cdf = CompositeDayFile().initFromParams(self._productType, 
                                                    bandName,
                                                    self._tid, 
                                                    year, 
                                                    day, 
                                                    self._compDir,
                                                    logger=self._logger,
                                                    dayDir=self._dayDir)

            raster = cdf.raster()
            cube[cubeIndex] = raster
            key = str(year) + str(day).zfill(3)
            xref[key] = cubeIndex
            cubeIndex += 1
            
        # cube = self._applyThreshold(cube)

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
    # ------------------------------------------------------------------------
    def getMetric(self, metricName: str) -> list[Metric]:
        
        self._logger.info('Getting ' + metricName)
        metricFileName = self._getOutName(metricName)
        metrics: list[Metric] = None
        
        if not metricFileName.exists():
            
            mFunc = self.availableMetrics[metricName]
            metrics = mFunc()
            self.writeMetric(metrics)

        else:
            metrics = self._readCache.get(metricName)

            if metrics is None:

                metrics = self.readMetric(metricFileName)
                self._readCache[metricName] = metrics
        
        return metrics
        
    # ------------------------------------------------------------------------
    # getMetricBand
    #
    # This is a bit of a mess from doing OO half way.  There is one metric,
    # UnsortedMonthlyBands, that is comprised of days.  When requesting that
    # metric, day must be specified or it will fail.  For all other metrics,
    # if day IS specified it will fail.
    # ------------------------------------------------------------------------
    def getMetricBand(self, 
                      metricName: str, 
                      bandName: str,
                      day: str = None) -> np.ndarray:
        
        metrics: list[Metric] = self.getMetric(metricName)
        soughtName = self._bandName(metricName, bandName, day)
        
        metric: Metric = [m for m in metrics if m.name == soughtName]
        
        if not metric:
            
            raise RuntimeError('Band ' + str(bandName) + 
                               ' not in ' + str(metricName))
                               
        assert len(metric) == 1
            
        return metric[0].value

    # ------------------------------------------------------------------------
    # _bandName
    #
    # The band name (the band's 'Name' metadata item) of a metric's band.
    # ------------------------------------------------------------------------
    def _bandName(self, metricName: str, bandName: str, day: str = None) -> str:

        soughtName = self._getBaseName(metricName)

        if bandName:

            bandName += '-Day_' + day if day is not None else ''
            soughtName += '-' + bandName

        return soughtName

    # ------------------------------------------------------------------------
    # _rfByFile
    #
    # RandomForestClassifier-format names grouped by metric file:
    # {metric: [(name, band name), ...]}, in the names' order.
    # ------------------------------------------------------------------------
    def _rfByFile(self, rfMetricNames: list) -> dict:

        byFile = {}

        for n in rfMetricNames:
            metric, band, day = self._rfParts(n)
            byFile.setdefault(metric, []).append((n, self._bandName(metric, band, day)))

        return byFile

    # ------------------------------------------------------------------------
    # _openBands
    #
    # Open a metric file for block reads of some of its bands: returns
    # (dataset, band numbers), the bands found by their 'Name' metadata item
    # (as readMetric names them).
    # ------------------------------------------------------------------------
    def _openBands(self, path: Path, metric: str, bandNames: list) -> tuple:

        if self._logger:
            self._logger.info('Reading blocks of ' + str(path))

        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        index = {ds.GetRasterBand(i + 1).GetMetadataItem('Name'): i + 1 for i in range(ds.RasterCount)}
        missing = [s for s in bandNames if s not in index]

        if missing:
            raise RuntimeError('Band ' + str(missing) + ' not in ' + str(metric))

        return ds, [index[s] for s in bandNames]

    # ------------------------------------------------------------------------
    # _readBlock
    #
    # Lines y0 to y1 - 1 of some bands of an open metric file, as int16 (as
    # readMetric gives them), shaped (bands, lines, columns).  The files are
    # pixel-interleaved, so one read gives every band of every pixel in the
    # block; GDAL's block cache is flushed afterwards (each block is read once).
    # ------------------------------------------------------------------------
    @staticmethod
    def _readBlock(ds, bandList: list, y0: int, y1: int) -> np.ndarray:

        block = ds.ReadAsArray(0, y0, ds.RasterXSize, y1 - y0, band_list=bandList)
        block = block.reshape(len(bandList), y1 - y0, ds.RasterXSize).astype(np.int16, copy=False)
        ds.FlushCache()
        return block

    # ------------------------------------------------------------------------
    # _readBlocks
    #
    # Read some bands of one metric file a block of lines at a time, without
    # holding the whole file: yields (y0, y1, block) (see _readBlock).
    # needLines(y0, y1) can skip a block.
    # ------------------------------------------------------------------------
    def _readBlocks(self, path: Path, metric: str, bandNames: list, linesPerRead: int = 256, needLines=None):

        ds, bandList = self._openBands(path, metric, bandNames)

        for y0 in range(0, ds.RasterYSize, linesPerRead):

            y1 = min(y0 + linesPerRead, ds.RasterYSize)

            if needLines is not None and not needLines(y0, y1):
                continue

            yield y0, y1, Metrics._readBlock(ds, bandList, y0, y1)

        ds = None

    # ------------------------------------------------------------------------
    # getRowsFromRf
    #
    # The values of RandomForestClassifier-format bands at given pixels only,
    # without holding whole metric files: each file is read once, a block of
    # lines at a time, and only the lines holding a pixel are read.  The
    # values equal getMetricFromRf(name)[xs, ys].  rowsByPart maps a part to
    # (xs, ys) = (raster rows, columns).  A metric file that does not exist
    # yet is made and read through getMetricFromRf.
    # ------------------------------------------------------------------------
    def getRowsFromRf(self, rfMetricNames: list, rowsByPart: dict, linesPerRead: int = 256) -> dict:

        out = {part: {} for part in rowsByPart}

        for metric, names in self._rfByFile(rfMetricNames).items():

            path = self._getOutName(metric)

            if not path.exists():

                for n, _ in names:
                    band = self.getMetricFromRf(n)
                    for part, (xs, ys) in rowsByPart.items():
                        out[part][n] = band[xs, ys]

                self._readCache.pop(metric, None)
                continue

            for part, (xs, ys) in rowsByPart.items():
                for n, _ in names:
                    out[part][n] = np.empty(len(xs), dtype=np.int16)

            allXs = np.concatenate([xs for xs, _ in rowsByPart.values()])
            need = lambda y0, y1: np.any((allXs >= y0) & (allXs < y1))

            for y0, y1, block in self._readBlocks(path, metric, [s for _, s in names], linesPerRead, need):

                for part, (xs, ys) in rowsByPart.items():
                    sel = np.flatnonzero((xs >= y0) & (xs < y1))
                    if len(sel):
                        for j, (n, _) in enumerate(names):
                            out[part][n][sel] = block[j, xs[sel] - y0, ys[sel]]

        return out

    # ------------------------------------------------------------------------
    # iterTableFromRf
    #
    # The full-tile predictor table of RandomForestClassifier-format names, a
    # block of lines at a time: yields (r0, r1, table rows r0 to r1 - 1).  Rows
    # are pixels (row-major, as ravel()), columns the distinct names in the
    # order of first appearance (as a DataFrame built column by column), in
    # dtype.  Every metric file is opened once and read a block of lines at a
    # time, so no whole file or whole table is held.  A metric file that does
    # not exist yet is made and read through getMetricFromRf.
    # ------------------------------------------------------------------------
    def iterTableFromRf(self, rfMetricNames: list, dtype=np.float32, linesPerRead: int = 256):

        columns = list(dict.fromkeys(rfMetricNames))
        col = {n: j for j, n in enumerate(columns)}
        rows, cols = self._productType.ROWS, self._productType.COLS
        files, arrays = [], []

        for metric, names in self._rfByFile(columns).items():

            path = self._getOutName(metric)

            if not path.exists():

                arrays += [(col[n], self.getMetricFromRf(n)) for n, _ in names]
                self._readCache.pop(metric, None)
                continue

            ds, bandList = self._openBands(path, metric, [s for _, s in names])

            if ds.RasterXSize != cols or ds.RasterYSize != rows:
                raise RuntimeError(str(path) + ': ' + str((ds.RasterYSize, ds.RasterXSize)) + ', expected ' + str((rows, cols)))

            files.append((ds, bandList, [col[n] for n, _ in names]))

        for y0 in range(0, rows, linesPerRead):

            y1 = min(y0 + linesPerRead, rows)
            table = np.empty(((y1 - y0) * cols, len(columns)), dtype=dtype)

            for ds, bandList, idx in files:

                block = Metrics._readBlock(ds, bandList, y0, y1)

                for j, c in enumerate(idx):
                    table[:, c] = block[j].ravel()

            for c, band in arrays:
                table[:, c] = band[y0:y1].ravel()

            yield y0 * cols, y1 * cols, table

        files = None

    # ------------------------------------------------------------------------
    # getTableFromRf
    #
    # The whole table of iterTableFromRf in one array.  It equals the
    # DataFrame of getMetricFromRf(name).ravel() columns, to_numpy(dtype).
    # Returns (table, column names).
    # ------------------------------------------------------------------------
    def getTableFromRf(self, rfMetricNames: list, dtype=np.float32, linesPerRead: int = 256) -> tuple:

        columns = list(dict.fromkeys(rfMetricNames))
        table = np.empty((self._productType.ROWS * self._productType.COLS, len(columns)), dtype=dtype)

        for r0, r1, part in self.iterTableFromRf(columns, dtype, linesPerRead):
            table[r0:r1] = part

        return table, columns

    # ------------------------------------------------------------------------
    # getMetricFromRf
    #
    # This returns a specific metric, band and day based on names passed in
    # RandomForestClassifier's format.  The requested yyyyddd can refer to
    # a model of a different year, so replace that yyyy with these metrics'
    # year, careful to adjust for the day wrapping to the next year.
    #
    # Special cases: 
    # UnsortedMonthlyBands-NDVI-Day_2020017
    # UnsortedMonthlyBands-Band_6-Day_2019289
    # TempMeanWarmest3
    # ------------------------------------------------------------------------
    def getMetricFromRf(self, rfMetricName: str) -> np.ndarray:

        return self.getMetricBand(*self._rfParts(rfMetricName))

    # ------------------------------------------------------------------------
    # _rfParts
    #
    # The metric, band and day that a RandomForestClassifier-format name
    # refers to (see getMetricFromRf).
    # ------------------------------------------------------------------------
    def _rfParts(self, rfMetricName: str) -> tuple:

        parts = rfMetricName.split('-')
        metric = 'metric' + parts[0]
        band = None
        day = None
        
        if len(parts) > 1:
            
            band = parts[1]
            day = parts[2].split('_')[1] if len(parts) == 3 else None

            # Map the model's yyyyddd to the year these metrics use for that DOY (year N, or N+1 for the days after the turn of
            # the year), so a model trained on one year's metrics can run on another year's.  Identity when the years match.
            if day is not None and len(day) == 7:
                day = str(self.getYearForDay(int(day[4:]))) + day[4:]
        
        return metric, band, day
        
    # ------------------------------------------------------------------------
    # getYearForDay
    # ------------------------------------------------------------------------
    def getYearForDay(self, day: int) -> int:
        
        year = [yd[0] for yd in self._daysSought if yd[1] == day][0]
        return year

    # ------------------------------------------------------------------------
    # getPeriodFlags
    #
    # The product's Quality and Cloud layers: one bit per composite period,
    # period 1 in bit 0, in the order of _daysSought.  Quality bit n is set
    # when period n fell back to the mean of all its input days (no day
    # passed QA); Cloud bit n when it fell back and none of its days was
    # clear.  The flags are the same for every band, so Band_1's are used,
    # read from the stored masks when present (no write access needed).
    # ------------------------------------------------------------------------
    def getPeriodFlags(self) -> (np.ndarray, np.ndarray):

        quality = np.zeros((self._productType.ROWS,
                            self._productType.COLS), dtype=np.uint8)

        cloud = np.zeros_like(quality)

        for n, (year, day) in enumerate(self._daysSought):

            cdf = CompositeDayFile().initFromParams(self._productType,
                                                    self._productType.BAND1,
                                                    self._tid,
                                                    year,
                                                    day,
                                                    self._compDir,
                                                    logger=self._logger,
                                                    dayDir=self._dayDir)

            qa, cl = cdf.getCompositeMasks()
            quality |= (qa.astype(np.uint8) & 1) << n
            cloud |= (cl.astype(np.uint8) & 1) << n

        return quality, cloud

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

            ndviCube = np.fromfile(outName).reshape(8, 
                                                    self._productType.ROWS,
                                                    self._productType.COLS)
                                                    
            return ndviCube, b1Xref

        # ---
        # Compute NDVI.
        # ---
        b2, b2Xref = self.getBandCube(self._productType.BAND2)

        # # ---
        # # Numpy division warnings are caused by NaNs and acceptable.
        # # When b1 == NaN or b2 == NaN, the desired result is NaN.
        # # ---
        # with np.errstate(divide = 'ignore', invalid = 'ignore'):
        #
        #     ndviRaw = np.where(b1 + b2 != 0, ((b2 - b1) / (b2 + b1)), 0)
        #     # ndviClamp = np.where(ndviRaw > 1, 1, ndviRaw)
        #     # ndviCube = ndviClamp * 1000
        #     ndviRaw = np.where(np.isnan(ndviRaw), 0, ndviRaw)
        #     ndviCube = ndviRaw * 1000
        #
        # ndviCube = self._applyThreshold(ndviCube)
        
        # ---
        # When both band 1 and band 2 have solz violations (and are assigned
        # a pixel value of 10000), NDVI becomes 0.  Instead of this behavior,
        # use the composite's mask to change solz violations to 10000 in the
        # NDVI output.
        # ---
        # ndviCube = np.where(b1.mask == True, 10000, ndviCube)

        # Use -1.001 to get -1001 when multiplied by 1000.
        ndviRaw = np.where(b1 + b2 != 0, ((b2 - b1) / (b2 + b1)), -1.001)
        
        # ---
        # LATER
        # zc = BandDayFile.ZENITH_CLAMP
        # np.where(b1 == zc == b2, -1.001, ndviRaw)
        # ---
        
        ndviCube = ndviRaw * 1000

        # VCF_ZENITH_NDVI=0 switches the zenith-only NDVI rule off (default on).
        if self._tid in Metrics.ZENITH_ONLY_NDVI_TILES and \
           os.environ.get('VCF_ZENITH_NDVI', '1') != '0':
            ndviCube = self._applyZenithOnlyNdvi(ndviCube)

        # Write NDVI.
        ndviCube.tofile(outName)

        # NDVI's xref is the same as band 1's.
        return ndviCube, b1Xref

    # ------------------------------------------------------------------------
    # applyZenithOnlyNdvi
    #
    # For each composite period, set NDVI to 0 where none of the period's
    # days has 0 < solz < ZENITH_CUTOFF.  A missing day file counts as no
    # valid zenith.  Every NDVI-derived metric is computed from the result.
    # ------------------------------------------------------------------------
    def _applyZenithOnlyNdvi(self, ndviCube: np.ndarray) -> np.ndarray:

        self._logger.info('Applying zenith-only NDVI to ' + self._tid)

        for i, (year, day) in enumerate(self._daysSought):

            cdf = CompositeDayFile().initFromParams(self._productType,
                                                    self._productType.BAND1,
                                                    self._tid,
                                                    year,
                                                    day,
                                                    self._compDir,
                                                    logger=self._logger,
                                                    dayDir=self._dayDir)

            valid = np.zeros((self._productType.ROWS,
                              self._productType.COLS), dtype=bool)

            for dYear, dDay in cdf._getDaysToFind():

                bdf = BandDayFile().initFromParams(self._productType,
                                                   self._productType.BAND1,
                                                   self._tid,
                                                   dYear,
                                                   dDay,
                                                   self._dayDir,
                                                   self._logger)
                try:
                    solz = bdf.getSolz()

                except RuntimeError as e:
                    self._logger.warning('No solz for ' + str(dYear) +
                                         str(dDay).zfill(3) + ': ' + str(e))
                    continue

                valid |= (solz > 0) & (solz < BandDayFile.ZENITH_CUTOFF)

            ndviCube[i] = np.where(valid, ndviCube[i], 0.0)

        return ndviCube

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
        
        ndvi, nXref = self.getNdvi()
        
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
    # readMetric
    # ------------------------------------------------------------------------
    def readMetric(self, metricPath: Path) -> list[Metric]:
        
        if self._logger:
            self._logger.info('Reading ' + str(metricPath))
            
        gif = GeospatialImageFile(str(metricPath))
        
        metrics: list[Metric] = []
        numBands = gif.getDataset().RasterCount
        
        for bandIndex in range(numBands):
            
            bandDs = gif.getDataset().GetRasterBand(bandIndex+1)
            value = bandDs.ReadAsArray().astype(np.int16)
            name = bandDs.GetMetadataItem('Name')
            desc = bandDs.GetMetadataItem('Description')
            metrics.append(Metrics.Metric(name, desc, value))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # writeMetric
    # ------------------------------------------------------------------------
    def writeMetric(self, metrics: list[Metric], name: str = None) -> None:

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

        modisSinusoidal = SpatialReference()
    
        modisSinusoidal.ImportFromProj4(
            '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 ' + 
            '+datum=WGS84 +units=m +no_defs')

        ds.SetSpatialRef(modisSinusoidal)
        outBandIndex = 0

        for metric in metrics:
            
            outBandIndex += 1
            gdBand = ds.GetRasterBand(outBandIndex)
            gdBand.WriteArray(metric.value)
            gdBand.SetMetadataItem('Name', metric.name)
            gdBand.SetMetadataItem('Description', metric.desc)
            # No per-band FlushCache: each flush rewrote the TIFF directory and
            # left stale copies in the file.  The dataset is written once when
            # closed; values, metadata and layout are unchanged.
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
        
        desc = 'unsorted monthly bands (1 – 7 and NDVI) = 96 metrics'
        self._logger.info('Running ' + desc)
            
        baseName = self._getMyBaseName()
        metrics = []
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:
            
            cube, dayXref = self.getBandCube(bandName)
            
            # ---
            # Each day is a metric.  This makes one huge file with each band's
            # days as individual metrics.
            # ---
            for day in dayXref:
                
                name = baseName + '-' + bandName + '-Day_' + str(day)
                desc = name.replace('-', ' ')
                index = dayXref[day]
                metrics.append(Metrics.Metric(name, desc, cube[index]))
                
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMax
    # ------------------------------------------------------------------------
    def metricBandReflMax(self) -> list:
        
        desc = 'maximum band X value (include NDVI) = 8 metrics'
        self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = self._getMyBaseName()
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            value = np.max(cube, axis=0)

            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMedian
    # ------------------------------------------------------------------------
    def metricBandReflMedian(self) -> list:
        
        desc = 'median band X value (include NDVI) = 8 metrics'
        self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = self._getMyBaseName()
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            value = np.median(cube, axis=0)
            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMin
    # ------------------------------------------------------------------------
    def metricBandReflMin(self) -> list[Metric]:
        
        desc = 'minimum band X value (include NDVI) = 8 metrics'
        self._logger.info('Running ' + desc)
            
        baseName = self._getMyBaseName()
        metrics = []

        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            value = cube.min(axis=0)

            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMaxGreenness
    # ------------------------------------------------------------------------
    def metricBandReflMaxGreenness(self) -> list:
        
        desc = 'band x reflectance associated with peak ' + \
               'greenness = 8 metrics'
               
        self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'BandReflMaxGreenness'
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByNDVI(cube)    
            value = sortedCube[-1, :, :]
            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMedianGreenness
    # ------------------------------------------------------------------------
    def metricBandReflMedianGreenness(self) -> list:

        desc = 'band x reflectance associated with median ' + \
               'greenness = 8 metrics'

        self._logger.info('Running ' + desc)

        metrics = []
        baseName = 'BandReflMedianGreenness'

        ndvi, nXref = self.getNdvi()

        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            # cNanToNdvi = np.where(np.isnan(cube), np.nan, ndvi)
            # ascIndexes = np.argsort(cNanToNdvi, axis=0)
            # sortedCube = np.take_along_axis(cube, ascIndexes, axis=0)
            # numNdviNotNan = (~np.isnan(ndvi)).sum(axis=0)  #2
            #
            # # This is considerably slower than Numpy functions.
            # value = np.empty((self._productType.ROWS, self._productType.COLS))
            #
            # for r in range(self._productType.ROWS):
            #
            #     for c in range(self._productType.COLS):
            #
            #         thisNnnn = numNdviNotNan[r, c]
            #         index = int(thisNnnn / 2)
            #         value[r, c] = sortedCube[index, r, c]

            sortedCube = self._sortByNDVI(cube)
            value = np.median(sortedCube, axis=0)
            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))

        return metrics

    # ------------------------------------------------------------------------
    # metricBandReflMinGreenness
    # ------------------------------------------------------------------------
    def metricBandReflMinGreenness(self) -> list:
        
        desc = 'band x reflectance associated with minimum ' + \
               'greenness = 8 metrics'
               
        self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'BandReflMinGreenness'
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByNDVI(cube, noDataLow=False)
            value = sortedCube[0, :, :]  # b/c sorted w/no-data high
            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))
        
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMaxTemp
    # ------------------------------------------------------------------------
    def metricBandReflMaxTemp(self) -> list:
        
        desc = 'band x reflectance associated with max ' + \
               'surface temp = 8 metrics'
               
        self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'BandReflMaxTemp'
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByThermal(cube)
            value = sortedCube[-1, :, :]  # b/c sorted in noDataLow order
                
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMedianTemp
    # ------------------------------------------------------------------------
    def metricBandReflMedianTemp(self) -> list:
        
        desc = 'band x reflectance associated with median ' + \
               'surf temp = 8 metrics'
               
        self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'BandReflMedianTemp'
        
        thermal, tXref = self.getBandCube(self._productType.BAND31)  # 1

        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            # cNanToTherm = np.where(np.isnan(cube), np.nan, thermal)
            # ascIndexes = np.argsort(cNanToTherm, axis=0)
            # sortedCube = np.take_along_axis(cube, ascIndexes, axis=0)
            # numThermNotNan = (~np.isnan(thermal)).sum(axis=0)  #2
            #
            # # This is considerably slower than Numpy functions.
            # value = np.empty((self._productType.ROWS, self._productType.COLS))
            #
            # for r in range(self._productType.ROWS):
            #
            #     for c in range(self._productType.COLS):
            #
            #         thisNtnn = numThermNotNan[r, c]
            #         index = int(thisNtnn / 2)
            #         value[r, c] = sortedCube[index, r, c]

            sortedCube = self._sortByThermal(cube)
            value = np.median(sortedCube, axis=0)
            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))

        return metrics
        
    # ------------------------------------------------------------------------
    # metricBandReflMinTemp
    # ------------------------------------------------------------------------
    def metricBandReflMinTemp(self) -> list:
        
        desc = 'band x reflectance associated with min ' + \
               'surface temp = 8 metrics'
               
        self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'BandReflMinTemp'
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByThermal(cube, noDataLow=False)
            value = sortedCube[0, :, :]
            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # lowestMeanBandRefl
    # ------------------------------------------------------------------------
    def _lowestMeanBandRefl(self, numBands: int) -> list:
        
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
            value = np.mean(slicedBand, axis=0)
            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))
            
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
        
        desc = 'Mean of ' + \
               str(numBands) + \
               ' greenest band x reflectance = 8 metrics'
               
        self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'Greenest' + str(numBands) + 'MeanBandRefl'
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByNDVI(cube)
            slicedBand = sortedCube[-numBands:, :, :]
            value = np.mean(slicedBand, axis=0)
            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))
            
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
        
        desc = 'Mean of ' + \
               str(numBands) + \
               ' warmest band x reflectance = 8 metrics'
               
        self._logger.info('Running ' + desc)
            
        metrics = []
        baseName = 'Warmest' + str(numBands) + 'MeanBandRefl'
        
        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByThermal(cube)
            slicedCube = sortedCube[-numBands:, :, :]
            value = np.nanmean(slicedCube, axis=0)
            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))
            
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
                             0,
                             maxBand - minBand)
                                   
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricAmpGreenestBandRefl
    # ------------------------------------------------------------------------
    def metricAmpGreenestBandRefl(self) -> list:
        
        desc = 'Sort band X by NDVI high to low:' + \
               ' amp == value 8 – value 1 = 8 metric'
               
        self._logger.info(desc)

        metrics = []
        baseName = 'AmpGreenestBandRefl'

        for bandName in self._productType.BANDS + [Metrics.NDVI]:

            cube, xref = self.getBandCube(bandName)
            sortedCube = self._sortByNDVI(cube)

            # # ---
            # # Nanargmin and nanargmax do not always work because it sometimes
            # # encounters all-nan slices and raises a value error.
            # #
            # # SortByNDVI() puts the band values related to NDVI NaN values
            # # at the beginning of the sorted array.
            # # ---
            # ndvi, ndviXref = self.getNdvi()
            # firstIndex = (np.isnan(ndvi).sum(axis=0))
            #
            # # Ensure the index is within the bounds.
            # firstIndex = np.where(firstIndex >= sortedCube.shape[0],
            #                       sortedCube.shape[0] - 1,
            #                       firstIndex)
            #
            # minBand = \
            #     np.take_along_axis(sortedCube,
            #                        firstIndex[None, :, :],
            #                        axis=0).reshape(firstIndex.shape)

            minBand = sortedCube[0]
            maxBand = sortedCube[-1]
            value = abs(maxBand - minBand)

            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))
            
        return metrics
        
    # ------------------------------------------------------------------------
    # metricAmpWarmestBandRefl
    # ------------------------------------------------------------------------
    def metricAmpWarmestBandRefl(self) -> list:
        
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
            cube, xref = self.getBandCube(bandName)

            # # ---
            # # Where the band is NaN, set thermal to NaN.  This prevents the
            # # minimum and maximum thermal values from corresponding to a NaN
            # # in the band, and limiting the amplitude value selection to
            # # valid values as much as possible.  This is effectively finding
            # # the extreme thermal indexes where the band is not NaN.
            # # ---
            # thermal, tXref = self.getBandCube(ProductType.BAND31)
            # thermal = np.where(np.isnan(cube), np.nan, thermal)
            #
            # # ---
            # # Using np.expand_dims() and np.squeeze() are necessary because
            # # our current version of Numpy, 1.21.5, does not have the
            # # keepdims argument for np.argmax() or np.argmin().
            # #
            # # Numpy propagates NaNs in min() and max().  Meanwhile nanmin()
            # # and nanmax() throw ValueError when encountering an all-NaN
            # # slice.  To work around all this, set NaNs to high values before
            # # finding the minimums, ...
            # # ---
            # thermalNansToHigh = np.where(np.isnan(thermal), np.inf, thermal)
            # minLocs = np.argmin(thermalNansToHigh, axis=0)
            # minLocs = np.expand_dims(minLocs, axis=0)
            # minValues = np.take_along_axis(cube, minLocs, axis=0).squeeze()
            #
            # # ---
            # # ...then set NaNs to low values before finding maximums.
            # # ---
            # thermalNansToLow = np.where(np.isnan(thermal), -np.inf, thermal)
            # maxLocs = np.argmax(thermalNansToLow, axis=0)
            # maxLocs = np.expand_dims(maxLocs, axis=0)
            # maxValues = np.take_along_axis(cube, maxLocs, axis=0).squeeze()

            sortedCube = self._sortByThermal(cube)
            minValues = sortedCube[0, :, :]
            maxValues = sortedCube[-1, :, :]
            value = abs(maxValues - minValues)
            
            name = baseName + '-' + bandName
            desc = name.replace('-', ' ')
            metrics.append(Metrics.Metric(name, desc, value))

        return metrics
        
    # ------------------------------------------------------------------------
    # metricTempMeanWarmest3
    # ------------------------------------------------------------------------
    def metricTempMeanWarmest3(self) -> list:
        
        desc = 'Mean temp of warmest 3 months = 1 metric'
        self._logger.info(desc)

        baseName = 'TempMeanWarmest3'        
        thermal, tXref = self.getBandCube(self._productType.BAND31)
        sortedCube = np.sort(thermal, axis=0)
        slicedArray = sortedCube[-3:, :, :]
        value = slicedArray.mean(axis=0).astype(int)
        metric = Metrics.Metric(baseName, desc, value)
        
        return [metric]
        
    # ------------------------------------------------------------------------
    # metricTempMeanGreenest3
    #
    # Temperature (aka B31)
    # Sorted by NDVI (aka Greenness)
    # Calculate the mean of the top 3 values (i.e. the greenest 3)
    #
    # Result will be a single band representing Temperature.
    # ------------------------------------------------------------------------
    def metricTempMeanGreenest3(self) -> list:
        
        desc = 'Mean temp of greenest 3 months = 1 metric'
        self._logger.info(desc)

        baseName = 'TempMeanGreenest3'
        thermal, tXref = self.getBandCube(self._productType.BAND31)
        sortedCube = self._sortByNDVI(thermal)

        # ---
        # There remains cases where there are fewer than three valid
        # values.  Create a masked array, masking the low values, take the
        # last three elements from the array, which may contain low values,
        # and compute the mean with masked_array.mean().
        # ---
        # mArray = ma.masked_equal(sortedCube, self._productType.NO_DATA)
        # slicedArray = mArray[-3:, :, :]
        # value = slicedArray.mean(axis=0).astype(int)
        # noDataValue = value.filled(self._productType.NO_DATA)

        slicedArray = sortedCube[-3:, :, :]
        value = slicedArray.mean(axis=0)

        metric = Metrics.Metric(baseName, desc, value)
        return [metric]


# ---
# C-mirror metrics: Python versions of the metrics C's make_metrics.c derives
# from the unsorted period values.  Installed here, after the class exists, so
# each family becomes one more metric method.  See CMirrorMetrics.py.
# ---
from modis_vcf.model import CMirrorMetrics  # noqa: E402
CMirrorMetrics.install(Metrics)


# ----------------------------------------------------------------------------
# Workers for Metrics._buildCompositesInParallel (module level, so the spawned
# processes can import them).
# ----------------------------------------------------------------------------
def _initCompositeWorker(level: int) -> None:
    logging.basicConfig(level=level, stream=sys.stdout)


def _buildComposite(productType: ProductType,
                    bandName: str,
                    tid: str,
                    year: int,
                    day: int,
                    compDir: Path,
                    dayDir: Path) -> None:

    CompositeDayFile().initFromParams(productType,
                                      bandName,
                                      tid,
                                      year,
                                      day,
                                      compDir,
                                      logger=logging.getLogger('composite'),
                                      dayDir=dayDir).raster()
