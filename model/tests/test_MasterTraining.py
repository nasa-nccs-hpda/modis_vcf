
import logging
from pathlib import Path
import sys
import tempfile
import unittest

from modis_vcf.model.MasterTraining import MasterTraining
from modis_vcf.model.TrainingType import TrainingType


# -----------------------------------------------------------------------------
# class MasterTrainingTestCase
#
# python -m unittest modis_vcf.model.tests.test_MasterTraining
# python -m unittest modis_vcf.model.tests.test_MasterTraining.MasterTrainingTestCase.testNumRows
# -----------------------------------------------------------------------------
class MasterTrainingTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUpClass
    # -------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):

        cls.logger = logging.getLogger()
        cls.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        cls.logger.addHandler(ch)

        cls.base = Path(__file__).parent
        
        cls.h08v04: Path = MasterTrainingTestCase.base / \
                           'h08v04-2019-pcttree-training+obsForRF.parq.skip'
        
        cls.h09v04: Path = MasterTrainingTestCase.base / \
                           'h09v04-2019-pcttree-training+obsForRF.parq'
        
        cls.h09v05: Path = MasterTrainingTestCase.base / \
                           'h09v05-2019-pcttree-training+obsForRF.parq'

        cls.validFragPaths = [MasterTrainingTestCase.h09v04,
                              MasterTrainingTestCase.h09v05]

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        tm = MasterTraining(MasterTrainingTestCase.base,
                            TrainingType.PCT_TREE,
                            MasterTrainingTestCase.logger)
                            
        # Ensure the expected fragments exist.
        self.assertEqual(len(tm._trainingDs.fragments), 2)
        
        self.assertTrue(Path(tm._trainingDs.fragments[0].path) in \
                        MasterTrainingTestCase.validFragPaths)

        self.assertTrue(Path(tm._trainingDs.fragments[1].path) in \
                        MasterTrainingTestCase.validFragPaths)

        # ---
        # Ensure the number of rows and columns are correct.
        # 259 columns - MasterTraining.startCol for each of 2 fragments.
        # ---
        self.assertEqual(len(tm._colNames), 8)
        self.assertEqual(len(tm._trainingDs.files), 2)

        # Test detection of a Parquet file with a mismatching schema.
        doNotSkipName = MasterTrainingTestCase.base / \
                        'h08v04-2019-pcttree-training+obsForRF.parq'
                        
        MasterTrainingTestCase.h08v04.rename(doNotSkipName)
        
        with self.assertRaisesRegex(RuntimeError, 'differs from the primary'):

            tm = MasterTraining(MasterTrainingTestCase.base,
                                TrainingType.PCT_TREE,
                                MasterTrainingTestCase.logger)
                                
        doNotSkipName.rename(MasterTrainingTestCase.h08v04)
        
        # Test invalid training directory.
        with self.assertRaisesRegex(RuntimeError, ' is invalid'):

            tm = MasterTraining(Path('invalid/dir'),
                                TrainingType.PCT_TREE,
                                MasterTrainingTestCase.logger)
        
        with self.assertRaisesRegex(RuntimeError, ' is invalid'):
            
            tm = MasterTraining(Path(__file__),  
                                TrainingType.PCT_TREE,
                                MasterTrainingTestCase.logger)
            
    # -------------------------------------------------------------------------
    # testDataset
    # -------------------------------------------------------------------------
    def testDataset(self):
        
        tm = MasterTraining(MasterTrainingTestCase.base,
                            TrainingType.PCT_TREE,
                            MasterTrainingTestCase.logger)

        self.assertEqual(len(tm.dataset.fragments), 2)
        
        self.assertTrue(Path(tm.dataset.fragments[0].path) in \
                        MasterTrainingTestCase.validFragPaths)

        self.assertTrue(Path(tm.dataset.fragments[1].path) in \
                        MasterTrainingTestCase.validFragPaths)

    # -------------------------------------------------------------------------
    # testToPandas
    # -------------------------------------------------------------------------
    def testToPandas(self):
        
        tm = MasterTraining(MasterTrainingTestCase.base,
                            TrainingType.PCT_TREE,
                            MasterTrainingTestCase.logger)

        df = tm.toPandas()
        self.assertEqual(df.shape[0], 536078)
        self.assertEqual(df.shape[1], 12)

    # -------------------------------------------------------------------------
    # testToCsv
    # -------------------------------------------------------------------------
    def testToCsv(self):
        
        tm = MasterTraining(MasterTrainingTestCase.base,
                            TrainingType.PCT_TREE,
                            MasterTrainingTestCase.logger)  
                            
        outDir = Path(tempfile.mkdtemp())
        csvPath = tm.writeCsv(outDir)
        print('CSV written to ' + str(csvPath))
        