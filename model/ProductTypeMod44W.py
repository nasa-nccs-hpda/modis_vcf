"""The MOD44W land / water mask (Collection 6.1, one file per tile and year: <dir>/<year>/001/MOD44W.A<year>001.<tile>.061.*.hdf).
"""


from pathlib import Path

import numpy as np

from modis_vcf.model.ProductType import ProductType


# ----------------------------------------------------------------------------
# Class ProductTypeMod44W
# ----------------------------------------------------------------------------
class ProductTypeMod44W(ProductType):
    """MOD44W as a product type: one band, water_mask (land, water, outside projection), one day per year."""
    WATER_MASK = 'water_mask'

    DAY_STEP = 365
    PRODUCT_TYPE = 'MOD44'
    W = 'W'
    YEAR_ONE_START_DAY = 1
    YEAR_ONE_END_DAY = 365
    YEAR_TWO_START_DAY = 0
    YEAR_TWO_END_DAY = 0
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, inputDir: Path):

        if inputDir is None:
            raise ValueError('No MOD44W input folder: set VCF_MOD44W_DIR.')

        super(ProductTypeMod44W, self).__init__( \
            ProductTypeMod44W.PRODUCT_TYPE, 
            inputDir,
            dayStep = ProductTypeMod44W.DAY_STEP,
            yearOneStartDay = ProductTypeMod44W.YEAR_ONE_START_DAY,
            yearOneEndDay = ProductTypeMod44W.YEAR_ONE_END_DAY,
            yearTwoStartDay = ProductTypeMod44W.YEAR_TWO_START_DAY,
            yearTwoEndDay = ProductTypeMod44W.YEAR_TWO_END_DAY)
        
        self._bandXref = {ProductTypeMod44W.WATER_MASK: 0}
        self._prefixXref = {ProductTypeMod44W.WATER_MASK: ProductTypeMod44W.W}
        self._productTypeXref = {ProductTypeMod44W.WATER_MASK: self}

    # ------------------------------------------------------------------------
    # createQaMask
    # ------------------------------------------------------------------------
    def createQaMask(self, 
                     state: np.ndarray,
                     solz: np.ndarray,
                     zenithCutOff: int) -> np.ndarray:
        """MOD44W has no QA: returns None."""
        return None

    # ------------------------------------------------------------------------
    # findFile
    # ------------------------------------------------------------------------
    def findFile(self, 
                 tileId: str, 
                 year: int, 
                 day: int,
                 bandName: str,
                 altDir: Path = None) -> Path:
        """The tile's MOD44W file of the year, from <inputDir>/<year>/001."""
        altDir: Path = \
            self.inputDir / Path(str(year)) / Path(str(day).zfill(3))
            
        return super(ProductTypeMod44W, self).findFile(tileId, 
                                                       year, 
                                                       day, 
                                                       bandName, 
                                                       altDir)
