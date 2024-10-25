
import numpy as np

from osgeo import gdal

from modis_vcf.model.DayFile import DayFile
from modis_vcf.model.ProductType import ProductType


# ----------------------------------------------------------------------------
# Class BandDayFile
# ----------------------------------------------------------------------------
class BandDayFile(DayFile):

    DEFAULT_ZENITH_CUTOFF = 72

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self):
        
        super(BandDayFile, self).__init__()

    # ------------------------------------------------------------------------
    # readSubdataset
    # ------------------------------------------------------------------------
    def _readSubdataset(self, 
                        inBandName: str = None, 
                        applyNoData: bool = True,
                        productType = None) -> (np.ndarray, int):
        
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
        bandNoData = bandDs.GetRasterBand(1).GetNoDataValue()
        bandDataType = bandDs.GetRasterBand(1).DataType
        self._geoTransform: tuple = bandDs.GetGeoTransform()
        
        # ReadAsArray automatically resamples when necessary.
        rawBand = bandDs.ReadAsArray(buf_xsize=ProductType.COLS,
                                     buf_ysize=ProductType.ROWS)
        
        if applyNoData:
            
            rawBand = \
                np.where(rawBand == bandNoData, ProductType.NO_DATA, rawBand)

        return (rawBand, bandDataType)

    # ------------------------------------------------------------------------
    # getRaster
    # ------------------------------------------------------------------------
    def _getRaster(self, applyQa: bool = True) -> np.ndarray:
        
        self._logger.info('Reading band from HDF')

        # Read the raster without QA.
        outBand, dataType = self._readSubdataset()
        
        # Apply the QA.  It does not use ProductType.NO_DATA.
        if applyQa:
            
            # ---
            # MOD09 uses the thermal band from MOD44.  When reading a band
            # from another product type, get the QA bands from that product
            # type, too.  ReadSubDataset() calls ProductType.findFile(), 
            # which looks up the product type mapping.  We must override this.
            # Self._productType will be PTMOD09.
            # ---
            productType = self._productType
            bandPt = self._productType.getProductTypeForBand(self.bandName)

            if bandPt != self._productType:
                productType = bandPt
            
            solz, dType = self._readSubdataset(ProductType.SOLZ,
                                               productType=productType)
        
            solz = (solz * self._productType.solarZenithScaleFactor). \
                   astype(np.int16)
               
            state, dtype = self._readSubdataset(ProductType.STATE, 
                                                False,
                                                productType=productType)
        
            self._qaMask: np.ndarray = self._productType. \
                createQaMask(state, 
                             solz,
                             BandDayFile.DEFAULT_ZENITH_CUTOFF)

            outBand = np.where(self._qaMask==1, outBand, ProductType.NO_DATA)
        
        # outBand = outBand.astype(np.int16)
        outBand.tofile(self.outName)
    
        return outBand

    # ------------------------------------------------------------------------
    # myName
    # ------------------------------------------------------------------------
    def _myName(self) -> str:
        return 'Band'
        