
from pathlib import Path
import unittest

from modis_vcf.model.VcfProcess import VcfProcess


# -----------------------------------------------------------------------------
# class VcfProcessTestCase
# -----------------------------------------------------------------------------
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_VcfProcess
# python -m unittest modis_vcf.model.tests.test_VcfProcess.VcfProcessTestCase.testInit
# -----------------------------------------------------------------------------
class VcfProcessTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUp
    # -------------------------------------------------------------------------
    def setUp(self):

        self._years = [2019]
        self._tids = ['h09v05']
        
        self._rfFile = Path('/explore/nobackup/people/rlgill/SystemTesting' +    
                            '/modis-vcf/MOD44/h09v05-2019-model.bin')
        
        self._outDir = Path('/explore/nobackup/people/rlgill/SystemTesting' +
                            '/modis-vcf/MOD44/vcfProcess')
        
        self._vcfp = VcfProcess(self._years, 
                                self._tids, 
                                self._rfFile,
                                self._outDir)
        
    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        vcfp = VcfProcess(self._years, self._tids, self._rfFile, self._outDir)
        self.assertEqual(vcfp._outDir, self._outDir)
        
    # -------------------------------------------------------------------------
    # testGetMetrics
    # -------------------------------------------------------------------------
    def testGetMetrics(self):
        
        self._vcfp._getMetrics(self._tids[0], self._years[0])

    # -------------------------------------------------------------------------
    # testGetUnsortedMonthlyBands
    # -------------------------------------------------------------------------
    def testGetUnsortedMonthlyBands(self):
        
        met = self._vcfp._getMetrics(self._tids[0], 
                                     self._years[0], 
                                     'UnsortedMonthlyBands-NDVI-Day_2020017')
                                    
        self.assertEqual(len(met), 1)

    # -------------------------------------------------------------------------
    # testRunOneTile
    # -------------------------------------------------------------------------
    def testRunOneTile(self):
        
        self._vcfp.runOneTile(self._tids[0], self._years[0])