
import unittest

from modis_vcf.model.DayFile import DayFile


# -----------------------------------------------------------------------------
# class DayFileTestCase
#
# pytest tests/test_DayFile.py
# pytest tests/test_DayFile.py::DayFileTestCase::testInit
# -----------------------------------------------------------------------------
class DayFileTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        with self.assertRaisesRegex(TypeError, 'instantiate abstract class'):
            df = DayFile()
        
