"""The base class of the input product types (ProductTypeMod44, ProductTypeMod44W)."""


from abc import ABC
from abc import abstractmethod
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# Class ProductType
#
# TODO: Complete validation in __init__().
# TODO: Refactor QA code as appropriate.
# TODO: Make product types register themselves.
# TODO: Remove or rename variables referring to no-data values.  Every pixel
#       must have some value for machine learning.
# ----------------------------------------------------------------------------
class ProductType(ABC):
    """An input product: its band names, input folder and file search, band layout in the files, QA mask and the days of the
    processing year (DOY 065 of the year to DOY 049 of the next).  Implemented by ProductTypeMod44 (the MOD44CH / MOD44CQ
    16-day composites) and ProductTypeMod44W (the MOD44W land / water mask)."""
    BAND1 = 'Band_1'
    BAND2 = 'Band_2'
    BAND3 = 'Band_3'
    BAND4 = 'Band_4'
    BAND5 = 'Band_5'
    BAND6 = 'Band_6'
    BAND7 = 'Band_7'
    BAND31 = 'Band_31'
    BANDS = [BAND1, BAND2, BAND3, BAND4, BAND5, BAND6, BAND7, BAND31]
    NDVI = 'NDVI'

    SOLZ = 'SolarZenith'
    STATE = 'State'

    ROWS = 4800
    COLS = 4800

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
            raise ValueError('A valid input directory must be specified, not ' + str(inputDir) + '.')
            
        self._inputDir: Path = inputDir
        self._solzScaleFactor: float = solzScaleFactor
        self._bandXref: dict = None
        self._prefixXref: dict = None

        # Input file suffix, and dataset names for HDF5 inputs, which are
        # selected by name because their subdataset order differs.
        self._fileSuffix: str = '.hdf'
        self._h5NameXref: dict = None
        self._productType: str = productType

        # {band name: the product type whose files hold it} (getProductTypeForBand).
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
        """{band name: subdataset index} in the product's files."""
        return self._bandXref

    # ------------------------------------------------------------------------
    # fileSuffix
    # ------------------------------------------------------------------------
    @property
    def fileSuffix(self) -> str:
        """The input file suffix: '.hdf' (HDF4) or '.h5' (HDF-EOS5, Collection 7 MOD44C)."""
        return self._fileSuffix

    # ------------------------------------------------------------------------
    # h5NameXref
    # ------------------------------------------------------------------------
    @property
    def h5NameXref(self) -> dict:
        """{band name: dataset name} for HDF-EOS5 inputs, read by name; None for HDF4 inputs."""
        return self._h5NameXref
        
    # ------------------------------------------------------------------------
    # createQaMask
    # ------------------------------------------------------------------------
    @abstractmethod
    def createQaMask(self, 
                     state: np.ndarray,
                     solz: np.ndarray,
                     zenithCutOff: int) -> np.ndarray:
        """The QA pass mask (1 = pass, int16) of one input day from its state word and solar zenith."""
        pass
        
    # ------------------------------------------------------------------------
    # dayStep
    # ------------------------------------------------------------------------
    @property
    def dayStep(self) -> int:
        """Days between input files (16 for the MOD44C composites)."""
        return self._dayStep
        
    # ------------------------------------------------------------------------
    # findFile
    # ------------------------------------------------------------------------
    def findFile(self, 
                 tileId: str, 
                 year: int, 
                 day: int,
                 bandName: str,
                 altDir: Path = None) -> Path:
        """The input file of a tile, year, day and band (the product type that holds the band, see getProductTypeForBand),
        searched in <inputDir>/<year> or in altDir; the first match.  Raises RuntimeError when there is none."""
        # The altDir argument is a kludge until this can be redesigned.
        searchPt: ProductType = self.getProductTypeForBand(bandName)
        globDir = altDir or searchPt._inputDir / str(year)

        prefix = searchPt._prefixXref[bandName]
        yNj = str(year) + str(day).zfill(3)
        pt = searchPt.productType
        mateGlob = pt + prefix + '.A' + yNj + '.' + tileId + '.*' + searchPt._fileSuffix
        mateFiles = list(globDir.glob(mateGlob))

        if not mateFiles:
            
            raise RuntimeError('Unable to find file for ' + 
                               mateGlob + 
                               ' in ' +
                               str(globDir))
            
        return mateFiles[0]
        
    # ------------------------------------------------------------------------
    # getProductTypeForBand
    # ------------------------------------------------------------------------
    def getProductTypeForBand(self, bandName: str):
        """The product type whose files hold bandName (a product type can take a band from another one)."""
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
        """The input folder (<inputDir>/<year>/...)."""
        return self._inputDir
        
    # ------------------------------------------------------------------------
    # productType
    # ------------------------------------------------------------------------
    @property
    def productType(self) -> str:
        """The product name used in file names (e.g. 'MOD44')."""
        return self._productType
        
    # -------------------------------------------------------------------------
    # solarZenithScaleFactor
    # -------------------------------------------------------------------------
    @property
    def solarZenithScaleFactor(self) -> float:
        """The factor that turns stored solar zenith into degrees."""
        return self._solzScaleFactor
        
    # ------------------------------------------------------------------------
    # yearOneDays
    # ------------------------------------------------------------------------
    @property
    def yearOneDays(self) -> list:
        """The input days of the processing year in its first calendar year (DOY 065 onwards)."""
        return self._yearOneDays

    # ------------------------------------------------------------------------
    # yearTwoDays
    # ------------------------------------------------------------------------
    @property
    def yearTwoDays(self) -> list:
        """The input days of the processing year in the next calendar year (up to DOY 049)."""
        return self._yearTwoDays
        
    # ------------------------------------------------------------------------
    # yearOneStartDay
    # ------------------------------------------------------------------------
    @property
    def yearOneStartDay(self) -> int:
        """First input day of the processing year (first calendar year)."""
        return self._yearOneStartDay

    # ------------------------------------------------------------------------
    # yearOneEndDay
    # ------------------------------------------------------------------------
    @property
    def yearOneEndDay(self) -> int:
        """End (exclusive) of the input days in the first calendar year."""
        return self._yearOneEndDay
        
    # ------------------------------------------------------------------------
    # yearTwoStartDay
    # ------------------------------------------------------------------------
    @property
    def yearTwoStartDay(self) -> int:
        """First input day in the next calendar year."""
        return self._yearTwoStartDay
        
    # ------------------------------------------------------------------------
    # yearTwoEndDay
    # ------------------------------------------------------------------------
    @property
    def yearTwoEndDay(self) -> int:
        """End (exclusive) of the input days in the next calendar year."""
        return self._yearTwoEndDay
        
