
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
    AERO_LOW = 64
    AERO_AVERAGE = 128
    CLOUD_ADJACENT = 8192
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
    # The MOD09A1 equivalent of the MOD44C rule
    # (ProductTypeMod44.createQaMask).  Pass a pixel only if:
    #
    #     cloud state (bits 0-1) is clear and internal cloud (bit 10) is not
    #     set (MOD44C bits 0-1 carry both)
    #     no cloud shadow (bit 2)
    #     not adjacent to cloud (bit 13; MOD44C bit 6)
    #     aerosol quantity (bits 6-7) is low or average (stands in for
    #     MOD44C bit 15, decided 2026-10-09: reject climatology and high)
    #     0 < solar zenith < zenithCutOff
    # ------------------------------------------------------------------------
    def createQaMask(self,
                     state: np.ndarray,
                     solz: np.ndarray,
                     zenithCutOff: int) -> np.ndarray:

        s = state.astype(np.int32)
        cloud = s & 3
        cloudInternal = s & ProductTypeMod09A.CLOUD_INT
        shadow = s & ProductTypeMod09A.CLOUD_SHADOW
        adjacency = s & ProductTypeMod09A.CLOUD_ADJACENT
        aerosol = s & ProductTypeMod09A.AERO_MASK

        mask = np.where((cloud == 0) &
                        (cloudInternal == 0) &
                        (shadow == 0) &
                        (adjacency == 0) &
                        ((aerosol == ProductTypeMod09A.AERO_LOW) |
                         (aerosol == ProductTypeMod09A.AERO_AVERAGE)) &
                        (solz > 0) &
                        (solz < zenithCutOff),
                        1,
                        0).astype(np.int16)

        return mask

    # ------------------------------------------------------------------------
    # createCloudMask
    #
    # Same test as ProductTypeMod44.createCloudMask (1 = not cloudy).
    # ------------------------------------------------------------------------
    def createCloudMask(self,
                        state: np.ndarray,
                        solz: np.ndarray,
                        zenithCutOff: int) -> np.ndarray:

        s = state.astype(np.int32)

        mask = np.where(((s & 3) == 0) &
                        ((s & ProductTypeMod09A.CLOUD_INT) == 0) &
                        (solz < zenithCutOff),
                        1,
                        0).astype(np.int16)

        return mask
