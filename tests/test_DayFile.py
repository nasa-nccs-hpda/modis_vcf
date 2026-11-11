
import unittest

from modis_vcf.model.DayFile import DayFile


# -----------------------------------------------------------------------------
# class DayFileTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_DayFile
# python -m unittest modis_vcf.model.tests.test_DayFile.DayFileTestCase.testInit
# -----------------------------------------------------------------------------
class DayFileTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        with self.assertRaisesRegex(TypeError, 'instantiate abstract class'):
            df = DayFile()
        
