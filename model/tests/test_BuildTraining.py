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
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44


# -----------------------------------------------------------------------------
# class BuildTrainingTestCase
#
# python -m unittest modis_vcf.model.tests.test_BuildTraining
# python -m unittest modis_vcf.model.tests.test_BuildTraining.BuildTrainingTestCase.testBuildDfRowDict   # noqa: E501
# -----------------------------------------------------------------------------
class BuildTrainingTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUpClass
    # -------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):

        cls._logger = logging.getLogger()
        cls._logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        cls._logger.addHandler(ch)

        cls._modDir = Path('/css/modis/Collection6.1/L3/MOD44B-VCF/dev')
        cls._metNames = ['metricTempMeanGreenest3']
        cls._productType = ProductTypeMod44(cls._modDir)
        cls._tids = ['h09v05']
        cls._trainingName = 'PercentTree'
        cls._year = 2019

        # Directories
        baseDir = Path('/explore/nobackup/people/rlgill/SystemTesting' +
                       '/modis-vcf/UnitTests')
                 
        cls._metDir = baseDir / '1-Metrics'
        cls._outDir = baseDir / '2-Training'
        cls._trainingDir = BuildTraining.TRAINING_DIR
                 
    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):

        # Test all valid input.
        bt = BuildTraining(BuildTrainingTestCase._year,
                           BuildTrainingTestCase._modDir,
                           BuildTrainingTestCase._metDir,
                           BuildTrainingTestCase._outDir,
                           BuildTrainingTestCase._trainingDir,
                           BuildTrainingTestCase._trainingName,
                           BuildTrainingTestCase._tids,
                           BuildTrainingTestCase._metNames,
                           BuildTrainingTestCase._productType,
                           BuildTrainingTestCase._logger)

        self.assertEqual(bt._year, BuildTrainingTestCase._year)
        self.assertEqual(bt._modisDir, BuildTrainingTestCase._modDir)
        self.assertEqual(bt._outDir, BuildTrainingTestCase._outDir)
        self.assertEqual(bt._logger, BuildTrainingTestCase._logger)
        self.assertEqual(bt._trainingName, BuildTrainingTestCase._trainingName)

        # Test invalid output directory.
        with self.assertRaisesRegex(RuntimeError, 'does not exist'):

            bt = BuildTraining(BuildTrainingTestCase._year,
                               BuildTrainingTestCase._modDir,
                               BuildTrainingTestCase._metDir,
                               Path('bogus'),
                               BuildTrainingTestCase._trainingDir,
                               BuildTrainingTestCase._trainingName,
                               BuildTrainingTestCase._tids,
                               BuildTrainingTestCase._metNames,
                               BuildTrainingTestCase._productType,
                               BuildTrainingTestCase._logger)

        # Test default tids.
        bt = BuildTraining(BuildTrainingTestCase._year,
                           BuildTrainingTestCase._modDir,
                           BuildTrainingTestCase._metDir,
                           BuildTrainingTestCase._outDir,
                           BuildTrainingTestCase._trainingDir,
                           BuildTrainingTestCase._trainingName,
                           None,
                           BuildTrainingTestCase._metNames,
                           BuildTrainingTestCase._productType,
                           BuildTrainingTestCase._logger)

        # Test default metric names.
        bt = BuildTraining(BuildTrainingTestCase._year,
                           BuildTrainingTestCase._modDir,
                           BuildTrainingTestCase._metDir,
                           BuildTrainingTestCase._outDir,
                           BuildTrainingTestCase._trainingDir,
                           BuildTrainingTestCase._trainingName,
                           BuildTrainingTestCase._tids,
                           None,
                           BuildTrainingTestCase._productType,
                           BuildTrainingTestCase._logger)

        # Test default product type.
        bt = BuildTraining(BuildTrainingTestCase._year,
                           BuildTrainingTestCase._modDir,
                           BuildTrainingTestCase._metDir,
                           BuildTrainingTestCase._outDir,
                           BuildTrainingTestCase._trainingDir,
                           BuildTrainingTestCase._trainingName,
                           BuildTrainingTestCase._tids,
                           BuildTrainingTestCase._metNames,
                           None,
                           BuildTrainingTestCase._logger)

        # Test default logger.
        bt = BuildTraining(BuildTrainingTestCase._year,
                           BuildTrainingTestCase._modDir,
                           BuildTrainingTestCase._metDir,
                           BuildTrainingTestCase._outDir,
                           BuildTrainingTestCase._trainingDir,
                           BuildTrainingTestCase._trainingName,
                           BuildTrainingTestCase._tids,
                           BuildTrainingTestCase._metNames,
                           BuildTrainingTestCase._productType)
                           
    # -------------------------------------------------------------------------
    # testGetTileIds
    # -------------------------------------------------------------------------
    def testGetTileIds(self):

        bt = BuildTraining(BuildTrainingTestCase._year,
                           BuildTrainingTestCase._modDir,
                           BuildTrainingTestCase._metDir,
                           BuildTrainingTestCase._outDir,
                           BuildTrainingTestCase._trainingDir,
                           BuildTrainingTestCase._trainingName,
                           BuildTrainingTestCase._tids,
                           BuildTrainingTestCase._metNames,
                           BuildTrainingTestCase._productType,
                           BuildTrainingTestCase._logger)

        tids = bt._getTileIds()
        self.assertIsNotNone(tids)
        self.assertIn(BuildTrainingTestCase._tids[0], tids)

    # -------------------------------------------------------------------------
    # testRun
    # -------------------------------------------------------------------------
    def testRun(self):

        bt = BuildTraining(BuildTrainingTestCase._year,
                           BuildTrainingTestCase._modDir,
                           BuildTrainingTestCase._metDir,
                           BuildTrainingTestCase._outDir,
                           BuildTrainingTestCase._trainingDir,
                           BuildTrainingTestCase._trainingName,
                           tileIds = ['h09v05'],
                           metricNames = ['metricAmpBandRefl'])

        trainingFiles: list = bt.run()
        self.assertIsNotNone(trainingFiles)

        parquetFile: Path = trainingFiles[0]
        print('Reading existing Parquet file, ' + str(parquetFile))
        df1: pd.DataFrame = pd.read_parquet(parquetFile)
        self.assertIsNotNone(df1)
        
        parquetFile.unlink()
        parquetFile: Path = bt.run()[0]
        df2: pd.DataFrame = pd.read_parquet(parquetFile)
        self.assertTrue(df1.equals(df2))
        