
import logging
import pathlib as Path
import re
import sys

# This is temporary, until every metric is a subclass of Metric.
from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.ProductType import ProductType
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44

import numpy as np


# ----------------------------------------------------------------------------
# TODO: Put in its own file, and rename.
# ----------------------------------------------------------------------------
from enum import Enum
class NewBand(Enum):
    
    BAND1 = 'BAND_1'
    BAND2 = 'BAND_2'
    BAND3 = 'BAND_3'
    BAND4 = 'BAND_4'
    BAND5 = 'BAND_5'
    BAND6 = 'BAND_6'
    BAND7 = 'BAND_7'
    ALL_BANDS = [BAND1, BAND2, BAND3, BAND4, BAND5, BAND6, BAND7]

# ----------------------------------------------------------------------------
# Class Metric
#
# This will become the abstract base class for the Template Method
# implementation.  Until then, it is operating as a Bridge pattern between
# this new Metric base class and the do-it-all Metrics class.
#
# This will need a Factory Method to instantiate subclasses.
#
# *** THIS IS PUT ON HOLD.  USE METRICS.getMetricFromRf() FOR NOW. ***
# ----------------------------------------------------------------------------
class Metric(object):

    # ------------------------------------------------------------------------
    # __init__
    #
    # A metric's value depends on its tile ID, year and product type; 
    # therefore these must initialize a metric.
    #
    # Name is provided by the subclass, when the new implementation is
    # complete.
    # ------------------------------------------------------------------------
    def __init__(self,
                 name: str,
                 tileId: str,
                 year: int,
                 productType: ProductType,
                 outDir: Path,
                 logger: logging.RootLogger = None,
                 nanThreshold = 9999):
        
        # ---
        # Is it a valid metric name?  The idea is to instantiate a valid
        # metric that the system can compute, then realize its value when
        # requested.
        #
        # Getting into a half-way hybrid mess with Metrics.availableMetrics.
        # This is solved by going to the full redesign.  At that point, the
        # factory will instantiate a Metric, and "name" will not be passed
        # to this constructor.
        # ---
        curName = name
        
        if curName not in dir(Metrics):
            
            curName = 'metric' + name
            
        if curName not in dir(Metrics):

                raise RuntimeError('Unknown metric, ' + 
                                   curName + 
                                   ', requested.')
                               
        self._name: str = curName
        
        # Validate band.
        # if not isinstance(band, NewBand):
        #     raise RuntimeError('Invalid band, ' + str(band))
        #
        # self._band: NewBand = band
        
        # Validate the output directory.
        if not outDir or not outDir.exists() or not outDir.is_dir():
            raise RuntimeError('A valid output directory must be provided.')
            
        self._outDir: Path = outDir
        
        # ---
        # These are not implemented as property setters because they are only
        # set during construction.
        # ---
        self._tid: str = self._setTid(tileId)
        self._year: int = self._setYear(year)
        
        if not productType:
            raise RuntimeError('A product type must be provided.')
            
        self._productType: ProductType = productType

        # From the Metrics invocation; later from the subclass construction.
        self._desc: str = None
        
        # This will probably be maintained after Metrics is gone.
        self._nanThreshold: int = nanThreshold

        # ---
        # These get values when the metric is realized.  The value will include
        # all days for the year; however, days will be realized only as needed.
        # So far, days only apply to the unsorted-monthly-band metric.
        # ---
        self._value: np.ndarray = None
        self._fileName: Path = None
        
        # Logger
        if not logger:
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            if (not logger.hasHandlers()):

                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                logger.addHandler(ch)

        self._logger: logging.RootLogger = logger

    # # ------------------------------------------------------------------------
    # # computeMetric
    # # ------------------------------------------------------------------------
    # def _computeMetric(self) -> np.ndarray:
    #
    #     # Computes and writes.  Part of the Template Method pattern.
    #     value: np.ndarray = self._myComputeMetric()
    #
    #     # Write the metric.
    #     metricFile: Path = self._write()
        
    # ------------------------------------------------------------------------
    # myComputeMetric
    # ------------------------------------------------------------------------
    def _myComputeMetric(self, 
                         bands: list = NewBand.ALL_BANDS,
                         day) -> np.ndarray:

        # ---
        # Template Method pattern guts go here in derived classes.  For now,
        # use Metrics.
        # 
        # Instantiate Metrics.  This will be obsolete when the subclasses are
        # implemented.  This would be better as a single instance in the 
        # constructor, but the aim is to keep it isolated and more like the
        # forthcoming template method.
        # ---
        oldMetrics = Metrics(self._tid,
                             self._year,
                             self._productType,
                             self._outDir,
                             self._logger,
                             self._nanThreshold)

        mFunc = oldMetrics.availableMetrics[self.name]
        allValues: list[Metrics.Metric] = mFunc()
        # self._desc = metric.desc
        
        # Within allValues, find the bands sought.
        for bandName in bands.value:

            nameToFind = (self.name + '-' + bandName).upper()
        
            for value in allValues:

                # ---
                # TODO:  When Metrics goes away, this will be a normal equality 
                # comparison.  Currently, _metricName is prefixed with "metric",
                # so Metrics understands it.
                # ---
                if nameToFind.endswith(value.name.upper()):

                    self._value = value.value
                    self._desc = value.desc

            if self._value is None:
                raise RuntimeError('Unable to compute metric.')
            
        return self._value

    # # ------------------------------------------------------------------------
    # # get
    # # ------------------------------------------------------------------------
    # def get(self) -> np.ndarray:
    #
    #     if self._value:
    #         return self._value
    #
    #     if self._fileName.exists():
    #         return self._read()
    #
    #     return self._computeMetric()

    # ------------------------------------------------------------------------
    # name
    # ------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name  

    # ------------------------------------------------------------------------
    # setTid
    # ------------------------------------------------------------------------
    def _setTid(self, value) -> str:

        # Validate the general format.
        rex = re.compile('^[h][0-1][0-9][v][0-3][0-9]', flags=re.IGNORECASE)
        match = rex.fullmatch(value)

        # Now validate the the ranges.  H: 0 - 35; V: 0 - 17
        if match:
            
            hVal = int(value[4:6])
            vVal = int(value[1:3])
            
            if hVal >= 0 and hVal <= 35 and vVal >=0 and vVal <= 17:
                return value
        
        raise ValueError('Invalid tile ID, ' + str(value))

    # ------------------------------------------------------------------------
    # setYear
    # ------------------------------------------------------------------------
    def _setYear(self, value) -> int:

        # The year 4k problem?
        if value >= 1999 and value < 4000:
            return value
            
        raise ValueError('Invalid year, ' + str(value))
            
    # ------------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------------
    # def _write(self) -> Path:
        
        
        