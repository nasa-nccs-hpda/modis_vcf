
from pathlib import Path

import numpy as np

from modis_vcf.model.ProductType import ProductType


# ----------------------------------------------------------------------------
# Class ProductTypeMod09G
#
# TODO: Unit test
# ----------------------------------------------------------------------------
class ProductTypeMod09G(ProductType):

    GA = 'GA'
    GQ = 'GQ'
    DAY_STEP = 1
    PRODUCT_TYPE = 'MOD09'
    YEAR_TWO_END_DAY = 0  # This will not necessarily wrap.
    
    AERO_CLIMATOLOGY = 0
    AERO_HIGH = 192
    AERO_MASK = 192
    CLOUDY = 1
    CLOUD_INT = 1024
    CLOUD_MIXED = 2
    CLOUD_SHADOW = 4
    SOLAR_ZENITH_SCALE_FACTOR = 0.01

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, inputDir: Path, auxInputDir: Path):
        
        super(ProductTypeMod09G, self). \
            __init__(ProductTypeMod09G.PRODUCT_TYPE, 
                     inputDir,
                     ProductTypeMod09G.SOLAR_ZENITH_SCALE_FACTOR,
                     dayStep = ProductTypeMod09G.DAY_STEP,
                     yearTwoEndDay = ProductTypeMod09G.YEAR_TWO_END_DAY)
          
        # These indices start at 0.  
        self._bandXref = {ProductType.BAND1: 1, ProductType.BAND2: 3,
                          ProductType.BAND3: 13, ProductType.BAND4: 14,
                          ProductType.BAND5: 15, ProductType.BAND6: 16,
                          ProductType.BAND7: 17, # ProductType.BAND31: 9,
                          ProductType.SOLZ: 5, ProductType.STATE: 1}

        self._prefixXref = {ProductType.BAND1: ProductTypeMod09G.GQ,
                            ProductType.BAND2: ProductTypeMod09G.GQ,
                            ProductType.BAND3: ProductTypeMod09G.GA,
                            ProductType.BAND4: ProductTypeMod09G.GA,
                            ProductType.BAND5: ProductTypeMod09G.GA,
                            ProductType.BAND6: ProductTypeMod09G.GA,
                            ProductType.BAND7: ProductTypeMod09G.GA,
                            ProductType.SOLZ: ProductTypeMod09G.GA,
                            ProductType.STATE: ProductTypeMod09G.GA}

        self._productTypeXref = \
            {ProductType.BAND1: self,
             ProductType.BAND2: self,
             ProductType.BAND3: self,
             ProductType.BAND4: self,
             ProductType.BAND5: self,
             ProductType.BAND6: self,
             ProductType.BAND7: self,
             ProductType.BAND31: self,
             ProductType.SOLZ: self,
             ProductType.STATE: self}
             
        self._auxInputDir: Path = auxInputDir

    # ------------------------------------------------------------------------
    # createQaMask
    #
    # Reject pixel if:
    #
    #     Cloud is set to mixed or cloudy
    #     Cloud shadow is set to “yes”
    #     Internal cloud is set to “yes”
    #     Aerosol is set to “high” or “climatology”
    #     Solar Zenith >72
    # ------------------------------------------------------------------------
    def createQaMask(self, 
                     state: np.ndarray,
                     solz: np.ndarray,
                     zenithCutOff: int) -> np.ndarray:
        
        cloudMixed = state & ProductTypeMod09G.CLOUD_MIXED
        cloudy = state & ProductTypeMod09G.CLOUDY
        shadow = state & ProductTypeMod09G.CLOUD_SHADOW
        cloudInternal = state & ProductTypeMod09G.CLOUD_INT
        aerosol = state & ProductTypeMod09G.AERO_MASK
        
        mask = np.where(
                        (cloudMixed == ProductTypeMod09G.CLOUD_MIXED) |
                        (cloudy == ProductTypeMod09G.CLOUDY) |
                        (shadow == ProductTypeMod09G.CLOUD_SHADOW) |
                        (cloudInternal == ProductTypeMod09G.CLOUD_INT) |
                        # (aerosol == ProductTypeMod09G.AERO_CLIMATOLOGY) |
                        (aerosol == ProductTypeMod09G.AERO_HIGH) |
                        (solz > zenithCutOff),
                        ProductType.NO_DATA,
                        1).astype(np.int16)

        return mask

    # ------------------------------------------------------------------------
    # findFile
    #
    # This is a kludge until this can be redesigned.
    # ------------------------------------------------------------------------
    def findFile(self, 
                 tileId: str, 
                 year: int, 
                 day: int,
                 bandName: str) -> Path:
        
        fName: Path = None
        
        try:
            fName = super().findFile(tileId, year, day, bandName)

        except RuntimeError:
            
            fName = super(). \
                    findFile(tileId, year, day, bandName, self._auxInputDir)
                    
        return fName
            