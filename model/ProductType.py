
from abc import ABC
from abc import abstractmethod
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# Class ProductType
#
# TODO: Complete validation in __init__().
# ----------------------------------------------------------------------------
class ProductType(ABC):

    BAND1 = 'Band_1'
    BAND2 = 'Band_2'
    BAND3 = 'Band_3'
    BAND4 = 'Band_4'
    BAND5 = 'Band_5'
    BAND6 = 'Band_6'
    BAND7 = 'Band_7'
    BAND31 = 'Band31'
    BANDS = [BAND1, BAND2, BAND3, BAND4, BAND5, BAND6, BAND7]
    NDVI = 'NDVI'

    SOLZ = 'SolarZenith'
    STATE = 'State'

    ROWS = 4800
    COLS = 4800
    NO_DATA = -10001

    YEAR_ONE_START_DAY = 65
    YEAR_ONE_END_DAY = 365
    YEAR_TWO_START_DAY = 1
    YEAR_TWO_END_DAY = 58

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    @abstractmethod
    def __init__(self, 
                 productType: str, 
                 inputDir: Path,
                 solzScaleFactor: float = 1.0,
                 dayStep: int = 8,
                 yearOneStartDay: int = YEAR_ONE_START_DAY,
                 yearOneEndDay: int = YEAR_ONE_END_DAY,
                 yearTwoStartDay: int = YEAR_TWO_START_DAY,
                 yearTwoEndDay: int = YEAR_TWO_END_DAY):

        if not productType:
            raise ValueError('A product type must be provided.')
        
        if not inputDir or not inputDir.exists() or not inputDir.is_dir():
            raise ValueError('A valid input directory must be specified.')
            
        self._inputDir: Path = inputDir
        self._solzScaleFactor: float = solzScaleFactor
        self._bandXref: dict = None
        self._prefixXref: dict = None
        self._productType: str = productType

        # MOD09 uses MOD44 for band 31.  Ensure the correct PT is chosen.
        self._productTypeXref: dict = None

        # Compositing data members. 
        self._dayStep: int = dayStep
        self._yearOneStartDay: int = yearOneStartDay 
        self._yearOneEndDay: int = yearOneEndDay 
        self._yearTwoStartDay: int = yearTwoStartDay 
        self._yearTwoEndDay: int = yearTwoEndDay 
        
        self._yearOneDays = list(range(self._yearOneStartDay,
                                       self._yearOneEndDay,
                                       self._dayStep))

        self._yearTwoDays = list(range(self._yearTwoStartDay,
                                       self._yearTwoEndDay,
                                       self._dayStep))
        
    # ------------------------------------------------------------------------
    # bandXref
    # ------------------------------------------------------------------------
    @property
    def bandXref(self) -> dict:
        return self._bandXref
        
    # ------------------------------------------------------------------------
    # createQaMask
    # ------------------------------------------------------------------------
    @abstractmethod
    def createQaMask(self, 
                     state: np.ndarray,
                     solz: np.ndarray,
                     zenithCutOff: int) -> np.ndarray:
        pass
        
    # ------------------------------------------------------------------------
    # dayStep
    # ------------------------------------------------------------------------
    @property
    def dayStep(self) -> int:
        return self._dayStep
        
    # ------------------------------------------------------------------------
    # findFile
    # ------------------------------------------------------------------------
    def findFile(self, 
                 tileId: str, 
                 year: int, 
                 day: int,
                 bandName: str) -> Path:
        
        searchPt: ProductType = self.getProductTypeForBand(bandName)
        globDir = searchPt._inputDir
        prefix = searchPt._prefixXref[bandName]
        yNj = str(year) + str(day).zfill(3)
        pt = searchPt.productType
        mateGlob = pt + prefix + '.A' + yNj + '.' + tileId + '.*.hdf'
        mateFiles = list(globDir.glob(mateGlob))

        if not mateFiles:
            raise RuntimeError('Unable to find file for ' + mateGlob)
            
        return mateFiles[0]
        
    # ------------------------------------------------------------------------
    # getProductTypeForBand
    #
    # MOD09 uses MOD44 for band 31.  Ensure the correct PT is chosen.
    # ------------------------------------------------------------------------
    def getProductTypeForBand(self, bandName: str):
        
        if bandName in self._productTypeXref:
            
            return self._productTypeXref[bandName]
            
        else:
            
            raise RuntimeError('Unable to determine ProductType for band ' + 
                               bandName)
        
    # ------------------------------------------------------------------------
    # inputDir
    # ------------------------------------------------------------------------
    @property
    def inputDir(self) -> Path:
        return self._inputDir
        
    # ------------------------------------------------------------------------
    # productType
    # ------------------------------------------------------------------------
    @property
    def productType(self) -> str:
        return self._productType
        
    # -------------------------------------------------------------------------
    # solarZenithScaleFactor
    # -------------------------------------------------------------------------
    @property
    def solarZenithScaleFactor(self) -> float:
        return self._solzScaleFactor
        
    # ------------------------------------------------------------------------
    # yearOneDays
    # ------------------------------------------------------------------------
    @property
    def yearOneDays(self) -> list:
        return self._yearOneDays

    # ------------------------------------------------------------------------
    # yearTwoDays
    # ------------------------------------------------------------------------
    @property
    def yearTwoDays(self) -> list:
        return self._yearTwoDays
        
    # ------------------------------------------------------------------------
    # yearOneStartDay
    # ------------------------------------------------------------------------
    @property
    def yearOneStartDay(self) -> int:
        return self._yearOneStartDay

    # ------------------------------------------------------------------------
    # yearOneEndDay
    # ------------------------------------------------------------------------
    @property
    def yearOneEndDay(self) -> int:
        return self._yearOneEndDay
        
    # ------------------------------------------------------------------------
    # yearTwoStartDay
    # ------------------------------------------------------------------------
    @property
    def yearTwoStartDay(self) -> int:
        return self._yearTwoStartDay
        
    # ------------------------------------------------------------------------
    # yearTwoEndDay
    # ------------------------------------------------------------------------
    @property
    def yearTwoEndDay(self) -> int:
        return self._yearTwoEndDay
        
