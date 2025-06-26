
from pathlib import Path
import unittest

import numpy as np

from modis_vcf.model.Band import Band
from modis_vcf.model.TrainingType import TrainingType
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
        
        basePath = Path('/explore/nobackup/people/rlgill/SystemTesting' +
                        '/modis-vcf/UnitTests/VcfPredict')
                            
        self._metricsDir = basePath / '1-Metrics'
        self._metricsDir.mkdir(parents=True, exist_ok=True)

        modelDir = basePath / '3-Models'
        modelDir.mkdir(parents=True, exist_ok=True)
        
        self._treeCoverRfFile = \
            basePath / '3-Models' / 'pcttree.bin'
        
        self._nonvegRfFile = \
            basePath / '3-Models' / 'pctbare.bin'
        
        self._outDir = basePath / '4-VcfProcess'
        self._outDir.mkdir(exist_ok=True)
        
        self._vcfp = VcfPredict(self._treeCoverRfFile, 
                                self._nonvegRfFile,
                                self._outDir, 
                                self._metricsDir)
        
    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        vcfp = VcfPredict(self._treeCoverRfFile, 
                          self._nonvegRfFile,
                          self._outDir, 
                          self._metricsDir)
                          
        self.assertEqual(vcfp._outDir, self._outDir)
        
    # -------------------------------------------------------------------------
    # testGetMask
    # -------------------------------------------------------------------------
    def testGetMask(self):
        
        mask = self._vcfp._getMask(self._tids[0], self._years[0])
        self.assertEqual(mask.dtype, np.int16)
        self.assertEqual(mask.shape, (4800, 4800))

    # -------------------------------------------------------------------------
    # testGetMetrics
    # -------------------------------------------------------------------------
    def testGetMetrics(self):
        
        self._vcfp._getMetrics(self._tids[0], 
                               self._years[0], 
                               self._vcfp._treeCoverRf.feature_names_in_)

    # -------------------------------------------------------------------------
    # testGetUnsortedMonthlyBands
    # -------------------------------------------------------------------------
    def testGetUnsortedMonthlyBands(self):
        
        met: pd.DataFrame = \
            self._vcfp._getMetrics(self._tids[0], 
                                   self._years[0], 
                                   ['UnsortedMonthlyBands-NDVI-Day_2020017'])
                                    
        self.assertEqual(met.shape, (23040000, 1))

        met: pd.DataFrame = \
            self._vcfp._getMetrics(self._tids[0], 
                                   self._years[0], 
                                   ['UnsortedMonthlyBands-Band_6-Day_2019289'])
                                    
    # -------------------------------------------------------------------------
    # testMaskPrediction
    # -------------------------------------------------------------------------
    def testMaskPrediction(self):
        
        X = self._vcfp._getMetrics(self._tids[0], 
                                   self._years[0],
                                   self._vcfp._treeCoverRf.feature_names_in_)
                                   
        X = X.replace(-10001, 10001)

        # Run Random Forest.
        prediction: np.ndarray = self._vcfp._treeCoverRf.predict(X). \
                                 astype(np.int16). \
                                 reshape(Band.ROWS, Band.COLS)

        maskedPred: np.ndarray = self._vcfp._maskPrediction(self._tids[1],
                                                            self._years[1],
                                                            prediction)
                                                            
        # Get mask.
        mask: np.ndarray = self._vcfp._getMask(self._tids[1], self._years[1])
        
        # Check water mask.
        self.assertTrue((mask == VcfPredict.MASK_WATER).any())
        self.assertTrue((maskedPred == VcfPredict.PRED_WATER).any())
        
        waterIndexes = np.nonzero(mask == VcfPredict.MASK_WATER)
        maskedPredAtWaterIndexes = maskedPred[waterIndexes]

        self.assertTrue((maskedPredAtWaterIndexes == \
                         VcfPredict.PRED_WATER).all())
        
        # Check out-of-projection mask.
        self.assertTrue((mask == VcfPredict.MASK_OUT_OF_PROJ).any())
        self.assertTrue((maskedPred == VcfPredict.PRED_OUT_OF_PROJ).any())
        
        oopIndexes = np.nonzero(mask == VcfPredict.MASK_OUT_OF_PROJ)
        maskedPredAtOopIndexes = maskedPred[oopIndexes]

        self.assertTrue((maskedPredAtOopIndexes == \
                         VcfPredict.PRED_OUT_OF_PROJ).all())
        
        # Ensure predictions are untouched where there is no mask.
        self.assertTrue((mask != VcfPredict.MASK_WATER).any())
        self.assertTrue((maskedPred != VcfPredict.PRED_WATER).any())
        self.assertTrue((mask != VcfPredict.MASK_OUT_OF_PROJ).any())
        self.assertTrue((maskedPred != VcfPredict.PRED_OUT_OF_PROJ).any())
 
        okIndexes = np.nonzero((mask != VcfPredict.MASK_WATER) & \
                               (mask != VcfPredict.MASK_OUT_OF_PROJ))
        
        maskedPredAtOkIndexes = maskedPred[okIndexes]
        
        self.assertTrue( \
            ((maskedPredAtOkIndexes != VcfPredict.PRED_WATER) &
             (maskedPredAtOkIndexes != VcfPredict.PRED_OUT_OF_PROJ)).all())
        
    # -------------------------------------------------------------------------
    # testRunTileForYear
    # -------------------------------------------------------------------------
    def testRunTileForYear(self):
        
        pctTree, pctNonVeg, pctNonTreeVeg = \
            self._vcfp.runTileForYear(self._tids[1], self._years[1])
            
        self.assertTrue((pctTree + pctNonVeg + pctNonTreeVeg < 100).all())
        self.assertTrue((pctTree != -10001).all())
        self.assertTrue((pctNonVeg != -10001).all())
        self.assertTrue((pctNonTreeVeg != -10001).all())

        pctTree, pctNonVeg, pctNonTreeVeg = \
            self._vcfp.runTileForYear(self._tids[0], self._years[0])

        self.assertTrue((pctTree + pctNonVeg + pctNonTreeVeg < 100).all())
        self.assertTrue((pctTree != -10001).all())
        self.assertTrue((pctNonVeg != -10001).all())
        self.assertTrue((pctNonTreeVeg != -10001).all())
        
    # -------------------------------------------------------------------------
    # testRun
    # -------------------------------------------------------------------------
    def testRun(self):

        self._vcfp.run([self._tids[0]], [self._years[0]])

        # Tile h11v02 is missing files, causing VCFP to fail. 
        self._vcfp.run([self._tids[1]], [self._years[1]])

        self._vcfp.run(self._tids, self._years)
        