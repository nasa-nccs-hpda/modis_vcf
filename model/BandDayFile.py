
import numpy as np

from osgeo import gdal

from modis_vcf.model.DayFile import DayFile
from modis_vcf.model.ProductType import ProductType

gdal.UseExceptions()


# ----------------------------------------------------------------------------
# Class BandDayFile
# ----------------------------------------------------------------------------
class BandDayFile(DayFile):

    ZENITH_CUTOFF = 70
    ZENITH_CLAMP = 16000

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

        # Read the raster without QA.
        outBand = self._readSubdataset()
        
        # Apply the QA.  It does not use ProductType.NO_DATA.
        if applyQa:
            
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
                np.where(solz <= 0, self.ZENITH_CLAMP, outBand)
               
            outBand = \
                np.where(solz > self.ZENITH_CUTOFF, self.ZENITH_CLAMP, outBand)
               
        # outBand.tofile(self.outName)
    
        return outBand

    # ------------------------------------------------------------------------
    # myName
    # ------------------------------------------------------------------------
    def _myName(self) -> str:
        return 'Band'
        