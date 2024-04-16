
import logging
from pathlib import Path
import sys
import tempfile
import unittest

from modis_vcf.model.MasterTraining import MasterTraining


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
        cls.h08v04 = MasterTrainingTestCase.base / 'h08v04-2019.parq.skip'
        cls.h09v04: Path = MasterTrainingTestCase.base / 'h09v04-2019.parq'
        cls.h09v05: Path = MasterTrainingTestCase.base / 'h09v05-2019.parq'

        cls.validFragPaths = [MasterTrainingTestCase.h09v04,
                              MasterTrainingTestCase.h09v05]

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        tm = MasterTraining(MasterTrainingTestCase.base,
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
        self.assertEqual(len(tm._colNames), 255)
        self.assertEqual(len(tm._trainingDs.files), 2)

        # Test detection of a Parquet file with a mismatching schema.
        doNotSkipName = MasterTrainingTestCase.base / 'h08v04-2019.parq'
        MasterTrainingTestCase.h08v04.rename(doNotSkipName)
        
        with self.assertRaisesRegex(RuntimeError, 'differs from the primary'):

            tm = MasterTraining(MasterTrainingTestCase.base,
                                MasterTrainingTestCase.logger)
                                
        doNotSkipName.rename(MasterTrainingTestCase.h08v04)
        
    # -------------------------------------------------------------------------
    # testDataset
    # -------------------------------------------------------------------------
    def testDataset(self):
        
        tm = MasterTraining(MasterTrainingTestCase.base,
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
                            MasterTrainingTestCase.logger)

        df = tm.toPandas()
        self.assertEqual(df.shape[0], 2000)
        self.assertEqual(df.shape[1], 259)

    # -------------------------------------------------------------------------
    # testToCsv
    # -------------------------------------------------------------------------
    def testToCsv(self):
        
        tm = MasterTraining(MasterTrainingTestCase.base,
                            MasterTrainingTestCase.logger)  
                            
        outDir = Path(tempfile.mkdtemp())
        csvPath = tm.writeCsv(outDir)
        print('CSV written to ' + str(csvPath))
        