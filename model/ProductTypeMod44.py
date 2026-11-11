
from pathlib import Path

import numpy as np

from modis_vcf.model.ProductType import ProductType
from modis_vcf.model.Settings import MOD44_COLLECTION


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

        if inputDir is None:
            raise ValueError('No MOD44CH / MOD44CQ input folder: set VCF_MOD44_DIR '
                             '(or pass the folder, e.g. vcfPredict.py --modisDir).')

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

        # Collection 7 composites are HDF-EOS5 files with the same dataset
        # names, values and state bits as Collection 6.1, in another order.
        if MOD44_COLLECTION == '7':

            self._fileSuffix = '.h5'

            self._h5NameXref = {ProductType.BAND1: 'Band_1', ProductType.BAND2: 'Band_2',
                                ProductType.BAND3: 'Band_3', ProductType.BAND4: 'Band_4',
                                ProductType.BAND5: 'Band_5', ProductType.BAND6: 'Band_6',
                                ProductType.BAND7: 'Band_7', ProductType.BAND31: 'Band_31',
                                ProductType.SOLZ: 'solar_zenith', ProductType.STATE: 'state'}

        elif MOD44_COLLECTION != '61':
            raise ValueError('VCF_MOD44_COLLECTION must be 61 or 7, not ' + MOD44_COLLECTION)

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
        adjacency = state & 64

        # Bit 15 is PGE72's overall QC flag (1 = good).  It replaces an
        # aerosol test on bits 7-8, which in MOD44C state overlap other
        # fields and rejected 29-50% of good pixels.  State is read as
        # int16, where bit 15 is the sign bit; widen first so the mask
        # also works under NumPy 2.
        qcGood = state.astype(np.int32) & 32768

        mask = np.where((cloud == 0) &
                        (shadow == 0) &
                        (qcGood != 0) &
                        (adjacency == 0) &
                        (solz > 0) &
                        (solz < zenithCutOff),
                        1,
                        0).astype(np.int16)

        return mask

    # ------------------------------------------------------------------------
    # createCloudMask
    # ------------------------------------------------------------------------
    def createCloudMask(self, 
                        state: np.ndarray,
                        solz: np.ndarray,
                        zenithCutOff: int) -> np.ndarray:
        
        cloud = state & 3

        mask = np.where((cloud == 0) &
                        (solz < zenithCutOff),
                        1,
                        0).astype(np.int16)

        return mask

    # ------------------------------------------------------------------------
    # getCloud
    # 
    # This should probably be in the base class, at least as an abstract
    # method.  Currently, it is part of an experiment related to 
    # CompositeDayFile.  Originally, there was the prospect of using multiple
    # types of MODIS data, necessitating ProductType.  This is evolving into
    # just one type, MOD44.
    # ------------------------------------------------------------------------
    def getCloud(self, state: np.ndarray) -> int:

        return state & 3
        
    # ------------------------------------------------------------------------
    # getShadow
    # ------------------------------------------------------------------------
    def getShadow(self, state: np.ndarray) -> int:

        return (state & 4) >> 2
                
    # ------------------------------------------------------------------------
    # getAdjacency
    # ------------------------------------------------------------------------
    def getAdjacency(self, state: np.ndarray) -> int:

        return (state & 8192) >> 13
                
    # ------------------------------------------------------------------------
    # getAerosol
    # ------------------------------------------------------------------------
    def getAerosol(self, state: np.ndarray) -> int:

        return (state & 192) >> 6                
