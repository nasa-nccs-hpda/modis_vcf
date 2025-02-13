
from pathlib import Path
import unittest

from modis_vcf.model.VcfPredict import VcfPredict


# -----------------------------------------------------------------------------
# class VcfPredictTestCase
# -----------------------------------------------------------------------------
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_VcfPredict
# python -m unittest modis_vcf.model.tests.test_VcfPredict.VcfPredictTestCase.testInit
# -----------------------------------------------------------------------------
class VcfPredictTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUp
    # -------------------------------------------------------------------------
    def setUp(self):

        self._years = [2019, 2020]
        self._tids = ['h09v05', 'h11v02']
        
        self._rfFile = Path('/explore/nobackup/people/rlgill/SystemTesting' + 
                            '/modis-vcf/MOD44-save/Test-MC-model.bin')
        
        self._outDir = Path('/explore/nobackup/people/rlgill/SystemTesting' +
                            '/modis-vcf/MOD44-save/vcfProcess')
        
        self._metricsDir = Path('/explore/nobackup/people/rlgill/' +
                                'SystemTesting/modis-vcf/MOD44-save')
        
        self._vcfp = VcfPredict(self._rfFile, self._outDir, self._metricsDir)
        
    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        vcfp = VcfPredict(self._rfFile, self._outDir, self._metricsDir)
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

        met = self._vcfp._getMetrics(self._tids[0], 
                                     self._years[0], 
                                     'UnsortedMonthlyBands-Band_6-Day_2019289')
                                    

    # -------------------------------------------------------------------------
    # testRunTileForYear
    # -------------------------------------------------------------------------
    def testRunTileForYear(self):
        
        self._vcfp.runTileForYear(self._tids[0], self._years[0])
        
    # -------------------------------------------------------------------------
    # testRun
    # -------------------------------------------------------------------------
    def testRun(self):
        
        self._vcfp.run(self._tids, self._years)
        