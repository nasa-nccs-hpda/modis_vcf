
from collections import namedtuple
from pathlib import Path

import numpy as np

from osgeo import gdal
from osgeo import gdal_array

from modis_vcf.model.Band import Band


# ----------------------------------------------------------------------------
# Class Mate
#
# TODO: accessors
# ----------------------------------------------------------------------------
class Mate(object):
    
    CH = 'CH'
    CQ = 'CQ'
    BandType = namedtuple('BandType', ('code', 'index'))
    BAND_TYPES = {CH: BandType(CH, 0), CQ: BandType(CQ, 1)}
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, fileName: Path = None):

        self._fileName: Path = None
        self._bandType: Mate.BandType = None
        self._dataset: gdal.Dataset = None

        if fileName:
            
            self.fileName = fileName
            self.bandType = Mate.discoverBandType(fileName)

    # ------------------------------------------------------------------------
    # discoverBandType
    # ------------------------------------------------------------------------
    @staticmethod
    def discoverBandType(fileName: Path) -> str:
        
        bType = fileName.name.split('.')[0].upper()[5:7]
        
        if not bType in Mate.BAND_TYPES:
            raise RuntimeError('Invalid band type for ' + str(fileName))
            
        return Mate.BAND_TYPES[bType]
        
    # ------------------------------------------------------------------------
    # getMyKey
    # ------------------------------------------------------------------------
    def getMyKey(self) -> str:
        
        return Mate.getKey(self.fileName)
        
    # ------------------------------------------------------------------------
    # getKey
    # ------------------------------------------------------------------------
    @staticmethod
    def getKey(fileName: Path) -> str:
        
        return fileName.name.split('.')[1][1:]
        
    # -------------------------------------------------------------------------
    # bandType
    # -------------------------------------------------------------------------
    @property
    def bandType(self) -> BandType:
        return self._bandType

    # -------------------------------------------------------------------------
    # bandType Setter
    # -------------------------------------------------------------------------
    @bandType.setter
    def bandType(self, bt: BandType) -> None:

        if not bt.code in self.BAND_TYPES:
            raise RuntimeError('Invalid band type, ' + bt.code)
            
        self._bandType = bt

    # -------------------------------------------------------------------------
    # fileName
    # -------------------------------------------------------------------------
    @property
    def fileName(self) -> Path:
        return self._fileName

    # -------------------------------------------------------------------------
    # fileName Setter
    # -------------------------------------------------------------------------
    @fileName.setter
    def fileName(self, name: Path) -> None:

        self._fileName = name
        
    # -------------------------------------------------------------------------
    # dataset
    # -------------------------------------------------------------------------
    @property
    def dataset(self) -> gdal.Dataset:
        
        if not self._dataset:
            self._dataset = gdal.Open(str(self.fileName))
            
        return self._dataset

    # ------------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------------
    def read(self, subdatasetIndex: int) -> (np.ndarray, int):
        
        bandDs = gdal.Open(self.dataset.GetSubDatasets()[subdatasetIndex][0])
        bandNoData = bandDs.GetRasterBand(1).GetNoDataValue()
        bandDataType = bandDs.GetRasterBand(1).DataType
        rawBand = bandDs.ReadAsArray()
        # rawBand = np.ma.masked_equal(rawBand, Band.NO_DATA)
        rawBand = np.where(rawBand == bandNoData, Band.NO_DATA, rawBand)

        return (rawBand, bandDataType)
        