
import numpy as np

from osgeo import gdal

from modis_vcf.model.DayFile import DayFile
from modis_vcf.model.ProductType import ProductType

gdal.UseExceptions()


# ----------------------------------------------------------------------------
# Class BandDayFile
# ----------------------------------------------------------------------------
class BandDayFile(DayFile):

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

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self):
        
        super(BandDayFile, self).__init__()

    # ------------------------------------------------------------------------
    # getSolz
    # ------------------------------------------------------------------------
    def getSolz(self, productType: ProductType = None) -> np.ndarray:
        
        # This supports MOD09 using the thermal band from MOD44.
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
        
        # This supports MOD09 using the thermal band from MOD44.
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
        subdatasetIndex: int = productType.bandXref[bandName]
        bandDs = gdal.Open(ds.GetSubDatasets()[subdatasetIndex][0])
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
        # zenith - confirmed via
        # Reports/Single_Pixel_Trace_h12v02_row50_col2264.txt. Matching
        # C's real behavior means Python must not substitute a value
        # here either.
        # ---
        outBand = self._readSubdataset()

        if self.bandName == ProductType.BAND31:

            # ---
            # MOD09 uses the thermal band from MOD44.  When reading a band
            # from another product type, get the QA bands from that product
            # type, too.  ReadSubDataset() calls ProductType.findFile(),
            # which looks up the product type mapping.  We must override this.
            # Self._productType will be ProductTypeMod09A (or G?).
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
        