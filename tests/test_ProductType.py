
import unittest

from modis_vcf.model.ProductType import ProductType


# -----------------------------------------------------------------------------
# class ProductTypeTestCase
#
# pytest tests/test_ProductType.py
# -----------------------------------------------------------------------------
class ProductTypeTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        with self.assertRaisesRegex(TypeError, 'instantiate abstract class'):
            pt = ProductType()
