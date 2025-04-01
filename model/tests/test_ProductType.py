
import unittest

from modis_vcf.model.ProductType import ProductType


# -----------------------------------------------------------------------------
# class ProductTypeTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_ProductType
# -----------------------------------------------------------------------------
class ProductTypeTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        with self.assertRaisesRegex(TypeError, 'instantiate abstract class'):
            pt = ProductType()
