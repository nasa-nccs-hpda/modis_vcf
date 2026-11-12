"""One band of one tile on one input day, read from the MOD44CH / MOD44CQ file (ProductTypeMod44), as PGE61's fix_bands.c
prepares it: the low-sun substitution (LowSun) on Band_1-7 and Band_31, and the Band_31 clamp above solar zenith 70.
"""

import os

import numpy as np

from osgeo import gdal

from modis_vcf.model.DayFile import DayFile
from modis_vcf.model.ProductType import ProductType
from modis_vcf.model import LowSun

gdal.UseExceptions()


# ----------------------------------------------------------------------------
# Class BandDayFile
# ----------------------------------------------------------------------------
class BandDayFile(DayFile):
    """A daily input band (DayFile).  Kept in memory by default; written to <outDir> only when writeDailies() is on."""
    ZENITH_CUTOFF = 65
    ZENITH_CLAMP = 16000

    # ---
    # fix_bands.c's zenith value clamp (">70 -> 16000") only actually
    # executes for Band_31: it's b=9 in fix_bands.c's band loop, processed
    # after b=7 (solar_zenith), by which point the array the clamp reads
    # is genuinely populated - unlike Bands 1-7 (b=0..6, processed before
    # b=7, where the same clamp is dead code, see _getRaster()'s comment).
    # Matching C's real behavior means applying this clamp to Band_31 only.
    # ---
    VALUE_CLAMP_ZENITH_CUTOFF = 70

    LOW_SUN_BANDS = (ProductType.BAND1, ProductType.BAND2, ProductType.BAND3,
                     ProductType.BAND4, ProductType.BAND5, ProductType.BAND6,
                     ProductType.BAND7, ProductType.BAND31)

    # ---
    # The dailies are kept in memory and not written to 1-Days, unless
    # VCF_WRITE_DAILIES=1 (--writeIntermediates).  Each daily is read once,
    # by the worker that builds its composite, so writing it only costs disk.
    # An environment variable, so that spawned composite workers see it.
    # ---
    WRITE_DAILIES_ENV = 'VCF_WRITE_DAILIES'

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self):

        super(BandDayFile, self).__init__()

    # ------------------------------------------------------------------------
    # writeDailies
    # ------------------------------------------------------------------------
    @staticmethod
    def setWriteDailies(write: bool) -> None:
        """Write the daily inputs to 1-Days (True) or keep them in memory only (False, the default), for this process and the
        composite workers it starts (through VCF_WRITE_DAILIES).
        """
        os.environ[BandDayFile.WRITE_DAILIES_ENV] = '1' if write else '0'

    @staticmethod
    def writeDailies() -> bool:
        """Whether daily inputs are written to 1-Days (setWriteDailies, VCF_WRITE_DAILIES=1)."""
        return os.environ.get(BandDayFile.WRITE_DAILIES_ENV, '0') == '1'

    def _writeCache(self) -> bool:
        return BandDayFile.writeDailies()

    # ------------------------------------------------------------------------
    # getSolz
    # ------------------------------------------------------------------------
    def getSolz(self, productType: ProductType = None) -> np.ndarray:
        """The day's solar zenith (degrees, scaled by the product type's factor), as int16."""
        # productType: read from another product type's file (a band borrowed from it).
        productType = productType or self._productType

        solz: np.ndarray = self._readSubdataset(ProductType.SOLZ,
                                                productType=productType)
                                    
        solz = (solz * self._productType.solarZenithScaleFactor). \
               astype(np.int16)
               
        return solz
        
    # ------------------------------------------------------------------------
    # getState
    # ------------------------------------------------------------------------
    def getState(self, productType: ProductType = None) -> np.ndarray:
        """The day's MOD44C state QA word, as stored (bit layout: ProductTypeMod44.createQaMask)."""
        # productType: read from another product type's file (a band borrowed from it).
        productType = productType or self._productType

        state: np.ndarray = self._readSubdataset(ProductType.STATE, 
                                                 productType=productType)

        return state
        
    # ------------------------------------------------------------------------
    # readSubdataset
    # ------------------------------------------------------------------------
    def _readSubdataset(self, 
                        inBandName: str = None, 
                        productType = None) -> np.ndarray:
        
        bandName = inBandName or self._bandName
        productType = productType or self._productType
        
        self._logger.info('Reading subdataset, ' + 
                          bandName +
                          ', using ' + 
                          productType.productType)
                          
        fileName: Path = productType.findFile(self._tid,
                                              self._year,
                                              self._day,
                                              bandName)
        
        ds: gdal.Dataset = gdal.Open(str(fileName))
        subdatasets: list = ds.GetSubDatasets()

        if fileName.suffix == '.h5':

            name = '/Data_Fields/' + productType.h5NameXref[bandName]
            matches = [s[0] for s in subdatasets if s[0].endswith(name)]

            if len(matches) != 1:

                raise RuntimeError('Expected one ' + name + ' in ' +
                                   str(fileName) + ', found ' +
                                   str(len(matches)))

            bandDs = gdal.Open(matches[0])

        else:

            subdatasetIndex: int = productType.bandXref[bandName]
            bandDs = gdal.Open(subdatasets[subdatasetIndex][0])

        self._geoTransform: tuple = bandDs.GetGeoTransform()
        
        # ReadAsArray automatically resamples when necessary.
        rawBand = bandDs.ReadAsArray(buf_xsize=ProductType.COLS,
                                     buf_ysize=ProductType.ROWS)
                                     
        return rawBand

    # ------------------------------------------------------------------------
    # getRaster
    # ------------------------------------------------------------------------
    def _getRaster(self, applyQa: bool = True) -> np.ndarray:

        self._logger.info('Reading band from HDF')

        # ---
        # fix_bands.c's zenith value clamp (">70 -> 16000") never actually
        # executes for Bands 1-7 in the real C pipeline: it reads the
        # solar-zenith array before that array is populated for these
        # bands (Bands 1-7 are processed before solar_zenith in
        # fix_bands.c's band loop). C's real, shipped value for these
        # bands is therefore always the raw reading, regardless of solar
        # zenith (confirmed by a single-pixel trace through the C code).
        # Matching C's real behavior means Python must not substitute a value
        # here either.
        # ---
        outBand = self._readSubdataset()

        # ---
        # Low-sun bright substitution (PGE61 fix_bands.c's intent; LE0).
        # In modes le0 (default) and s70, every Band_1-7 and Band_31 input
        # whose solar zenith is above 70 becomes ZENITH_CLAMP (16000); le0
        # also substitutes solz <= 0 (polar night).  Only the zenith test
        # triggers it; inputs failing other QA tests keep their values.
        # Mode off = v1.3.0 behaviour (Band_31 only, > 70).  See LowSun.
        # ---
        lowSunT, lowSunLe0 = LowSun.settings()

        if self.bandName in BandDayFile.LOW_SUN_BANDS and \
           LowSun.appliesTo(self._tid):

            productType = self._productType
            bandPt = self._productType.getProductTypeForBand(self.bandName)

            if bandPt != self._productType:
                productType = bandPt

            solz = self.getSolz(productType=productType)
            lowSun = LowSun.mask(solz, lowSunT, lowSunLe0)

            return np.where(lowSun, self.ZENITH_CLAMP, outBand)

        if self.bandName == ProductType.BAND31:

            # ---
            # A product type may take the thermal band from another product
            # type.  When reading a band from another product type, get the
            # QA bands from that product type, too.  ReadSubDataset() calls
            # ProductType.findFile(), which looks up the product type
            # mapping.  We must override this.
            # ---
            productType = self._productType
            bandPt = self._productType.getProductTypeForBand(self.bandName)

            if bandPt != self._productType:
                productType = bandPt

            solz = self.getSolz(productType=productType)

            outBand = \
                np.where(solz > self.VALUE_CLAMP_ZENITH_CUTOFF, self.ZENITH_CLAMP, outBand)

        return outBand

    # ------------------------------------------------------------------------
    # myName
    # ------------------------------------------------------------------------
    def _myName(self) -> str:
        return 'Band'
        