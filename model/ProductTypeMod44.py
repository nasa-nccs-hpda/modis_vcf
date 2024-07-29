
from pathlib import Path

import numpy as np

from modis_vcf.model.ProductType import ProductType


# ----------------------------------------------------------------------------
# Class ProductTypeMod44
#
# TODO: complete testBandXref
# ----------------------------------------------------------------------------
class ProductTypeMod44(ProductType):

    CH = 'CH'
    CQ = 'CQ'
    DAY_STEP = 16
    PRODUCT_TYPE = 'MOD44'
    YEAR_ONE_END_DAY = 354
    YEAR_TWO_END_DAY = 50
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, inputDir: Path):
        
        super(ProductTypeMod44, self). \
            __init__(ProductTypeMod44.PRODUCT_TYPE, 
                     inputDir,
                     dayStep = ProductTypeMod44.DAY_STEP,
                     yearOneEndDay = ProductTypeMod44.YEAR_ONE_END_DAY,
                     yearTwoEndDay = ProductTypeMod44.YEAR_TWO_END_DAY)
        
        # SOLZ was 3
        self._bandXref = {ProductType.BAND1: 4, ProductType.BAND2: 5,
                          ProductType.BAND3: 0, ProductType.BAND4: 1,
                          ProductType.BAND5: 2, ProductType.BAND6: 3,
                          ProductType.BAND7: 4, ProductType.BAND31: 9,
                          ProductType.SOLZ: 2, ProductType.STATE: 0}

        self._prefixXref = {ProductType.BAND1: ProductTypeMod44.CQ,
                            ProductType.BAND2: ProductTypeMod44.CQ,
                            ProductType.BAND3: ProductTypeMod44.CH,
                            ProductType.BAND4: ProductTypeMod44.CH,
                            ProductType.BAND5: ProductTypeMod44.CH,
                            ProductType.BAND6: ProductTypeMod44.CH,
                            ProductType.BAND7: ProductTypeMod44.CH,
                            ProductType.BAND31: ProductTypeMod44.CH,
                            ProductType.SOLZ: ProductTypeMod44.CQ,
                            ProductType.STATE: ProductTypeMod44.CQ}

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

    # ------------------------------------------------------------------------
    # createQaMask
    # ------------------------------------------------------------------------
    def createQaMask(self, 
                     state: np.ndarray,
                     solz: np.ndarray,
                     zenithCutOff: int) -> np.ndarray:
        
        cloud = state & 3
        shadow = state & 4
        adjacency = state & 8192
        aerosol = (state & 192) >> 6

        mask = np.where((cloud == 0) &
                        (shadow == 0) &
                        (aerosol != 3) &   
                        (adjacency == 0) & 
                        (solz < zenithCutOff),
                        1,
                        ProductType.NO_DATA).astype(np.int16)

        return mask
