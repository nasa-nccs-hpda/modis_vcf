
from pathlib import Path

import numpy as np

from modis_vcf.model.ProductType import ProductType
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44


# ----------------------------------------------------------------------------
# Class ProductTypeMod09A
#
# Currently, we are using MOD44's band 31 as the thermal band.  MOD44 is a
# different product type, which makes things messy, and ripples throughout
# the code.
#
# Band 5: 2400 x 2400, 16-bit integer ("3")
# ----------------------------------------------------------------------------
class ProductTypeMod09A(ProductType):

    A1 = 'A1'
    DAY_STEP = 8
    PRODUCT_TYPE = 'MOD09'
    YEAR_TWO_END_DAY = 58
    
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
        
        super(ProductTypeMod09A, self). \
            __init__(ProductTypeMod09A.PRODUCT_TYPE, 
                     inputDir,
                     ProductTypeMod09A.SOLAR_ZENITH_SCALE_FACTOR,
                     dayStep = ProductTypeMod09A.DAY_STEP,
                     yearTwoEndDay = ProductTypeMod09A.YEAR_TWO_END_DAY)
            
        self._bandXref = {ProductType.BAND1: 0, ProductType.BAND2: 1,
                          ProductType.BAND3: 2, ProductType.BAND4: 3,
                          ProductType.BAND5: 4, ProductType.BAND6: 5,
                          ProductType.BAND7: 6, ProductType.BAND31: 9,
                          ProductType.SOLZ: 8, ProductType.STATE: 11}

        self._prefixXref = {ProductType.BAND1: ProductTypeMod09A.A1,
                            ProductType.BAND2: ProductTypeMod09A.A1,
                            ProductType.BAND3: ProductTypeMod09A.A1,
                            ProductType.BAND4: ProductTypeMod09A.A1,
                            ProductType.BAND5: ProductTypeMod09A.A1,
                            ProductType.BAND6: ProductTypeMod09A.A1,
                            ProductType.BAND7: ProductTypeMod09A.A1,
                            ProductType.SOLZ: ProductTypeMod09A.A1,
                            ProductType.STATE: ProductTypeMod09A.A1}

        self._productTypeXref = \
            {ProductType.BAND1: self,
             ProductType.BAND2: self,
             ProductType.BAND3: self,
             ProductType.BAND4: self,
             ProductType.BAND5: self,
             ProductType.BAND6: self,
             ProductType.BAND7: self,
             ProductType.BAND31: ProductTypeMod44(auxInputDir),
             ProductType.SOLZ: self,
             ProductType.STATE: self}

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
        
        cloudMixed = state & ProductTypeMod09A.CLOUD_MIXED
        cloudy = state & ProductTypeMod09A.CLOUDY
        shadow = state & ProductTypeMod09A.CLOUD_SHADOW
        cloudInternal = state & ProductTypeMod09A.CLOUD_INT
        aerosol = state & ProductTypeMod09A.AERO_MASK
        
        mask = np.where((cloudMixed == ProductTypeMod09A.CLOUD_MIXED) |
                        (cloudy == ProductTypeMod09A.CLOUDY) |
                        (shadow == ProductTypeMod09A.CLOUD_SHADOW) |
                        (cloudInternal == ProductTypeMod09A.CLOUD_INT) |
                        (aerosol == ProductTypeMod09A.AERO_CLIMATOLOGY) |
                        (aerosol == ProductTypeMod09A.AERO_HIGH) |
                        (solz > zenithCutOff),
                        ProductType.NO_DATA,
                        1).astype(np.int16)

        return mask
        