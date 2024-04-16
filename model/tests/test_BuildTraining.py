import logging
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

from modis_vcf.model.Band import Band
from modis_vcf.model.BuildTraining import BuildTraining
from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.Pair import Pair


# -----------------------------------------------------------------------------
# class BuildTrainingTestCase
#
# python -m unittest modis_vcf.model.tests.test_BuildTraining
# python -m unittest modis_vcf.model.tests.test_BuildTraining.BuildTrainingTestCase.testBuildDfRowDict
# -----------------------------------------------------------------------------
class BuildTrainingTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUpClass
    # -------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):

        cls._samplesFile = Path(__file__).parent / 'h09v05.samp.bin'
        cls._testDfFile = Path(__file__).parent / 'test.parquet'
        cls._tid = 'h09v05'
        cls._trainingName = 'PercentTree'

        cls._logger = logging.getLogger()
        cls._logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        cls._logger.addHandler(ch)

        cls._modDir = Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')
        cls._year = 2019
        
        # ---
        # This caches the bands and metric files, so subsequent tests run
        # much faster.  Certainly use your own directory.
        # ---
        cls._outDir = Path('/explore/nobackup/people/rlgill' +      
                           '/SystemTesting/modis-vcf')
        
    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        bt = BuildTraining(BuildTrainingTestCase._year,
                           BuildTrainingTestCase._modDir,
                           BuildTrainingTestCase._outDir,
                           BuildTrainingTestCase._logger,
                           BuildTrainingTestCase._trainingName)

        self.assertEqual(bt._year, BuildTrainingTestCase._year)
        self.assertEqual(bt._modisDir, BuildTrainingTestCase._modDir)
        self.assertEqual(bt._outDir, BuildTrainingTestCase._outDir)
        self.assertEqual(bt._logger, BuildTrainingTestCase._logger)
        self.assertEqual(bt._trainingName, BuildTrainingTestCase._trainingName)

        with self.assertRaisesRegex(RuntimeError, 'does not exist'):

            bt = BuildTraining(BuildTrainingTestCase._year,
                               BuildTrainingTestCase._modDir,
                               Path('bogus'),
                               BuildTrainingTestCase._logger,
                               BuildTrainingTestCase._trainingName)
        
    # -------------------------------------------------------------------------
    # testGetTileIds
    # -------------------------------------------------------------------------
    def testGetTileIds(self):

        tids = BuildTraining._getTileIds()
        self.assertIsNotNone(tids)
        self.assertIn(BuildTrainingTestCase._tid, tids)

    # -------------------------------------------------------------------------
    # testRun
    # -------------------------------------------------------------------------
    def testRun(self):

        bt = BuildTraining(BuildTrainingTestCase._year,
                           BuildTrainingTestCase._modDir,
                           BuildTrainingTestCase._outDir,
                           BuildTrainingTestCase._logger,
                           BuildTrainingTestCase._trainingName,
                           ['h08v04', 'h08v05', 'h09v04', 'h09v05'])

        parquetFile = bt.getParquetName()
        
        if not parquetFile.exists():
            
            parquetFile = bt.run()

        else:

            print('Reading existing Parquet file, ' + str(parquetFile))
            restoredDf = pd.read_parquet(parquetFile)
