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
        self.assertEqual(bt.trainingName, BuildTrainingTestCase._trainingName)

        with self.assertRaisesRegex(RuntimeError, 'does not exist'):

            bt = BuildTraining(BuildTrainingTestCase._year,
                               BuildTrainingTestCase._modDir,
                               Path('bogus'),
                               BuildTrainingTestCase._logger)
        
    # -------------------------------------------------------------------------
    # testBuildDfRowDict
    # -------------------------------------------------------------------------
    def testBuildDfRowDict(self):

        bt = BuildTraining(BuildTrainingTestCase._year,
                           BuildTrainingTestCase._modDir,
                           BuildTrainingTestCase._outDir,
                           BuildTrainingTestCase._logger,
                           BuildTrainingTestCase._trainingName)

        d = bt._buildDfRowDict([BuildTrainingTestCase._tid])
        self.assertEqual(d['h09v05-0-0'], ['h09v05', 0, 0])
        self.assertEqual(len(d), 4800 * 4800)
        
    # -------------------------------------------------------------------------
    # testGetTileIds
    # -------------------------------------------------------------------------
    def testGetTileIds(self):

        tids = BuildTraining._getTileIds(BuildTrainingTestCase._modDir,
                                         BuildTrainingTestCase._year)

        self.assertIsNotNone(tids)
        self.assertIn(BuildTrainingTestCase._tid, tids)

    # -------------------------------------------------------------------------
    # testMetricToCols
    # -------------------------------------------------------------------------
    def testMetricToCols(self):

        bt = BuildTraining(BuildTrainingTestCase._year,
                           BuildTrainingTestCase._modDir,
                           BuildTrainingTestCase._outDir,
                           BuildTrainingTestCase._logger,
                           BuildTrainingTestCase._trainingName)

        # Read the test data frame containing only tid, x, y.
        df = pd.read_parquet(BuildTrainingTestCase._testDfFile)
        
        # Get a band to add.
        mets = Metrics(BuildTrainingTestCase._tid,
                       BuildTrainingTestCase._year,
                       BuildTrainingTestCase._modDir,
                       BuildTrainingTestCase._outDir,
                       BuildTrainingTestCase._logger)

        band = mets.getBand(Pair.BAND1)
        
        # Add the band to the data frame.
        bt._metricToCols(df, band)
        

    # -------------------------------------------------------------------------
    # testProcessTid
    # -------------------------------------------------------------------------
    def testProcessTid(self):

        bt = BuildTraining(BuildTrainingTestCase._year,
                           BuildTrainingTestCase._modDir,
                           BuildTrainingTestCase._outDir,
                           BuildTrainingTestCase._logger,
                           BuildTrainingTestCase._trainingName)

        df = pd.read_parquet(BuildTrainingTestCase._testDfFile)
        bt._processTid(df, BuildTrainingTestCase._tid)
        
    # -------------------------------------------------------------------------
    # testRun
    # -------------------------------------------------------------------------
    def testRun(self):

        bt = BuildTraining(BuildTrainingTestCase._year,
                           BuildTrainingTestCase._modDir,
                           BuildTrainingTestCase._outDir,
                           BuildTrainingTestCase._logger,
                           BuildTrainingTestCase._trainingName)

        parquetFile = bt.getParquetName()
        
        if not parquetFile.exists():
            
            parquetFile = bt.run(BuildTrainingTestCase._tid)

        else:

            print('Reading existing Parquet file, ' + str(parquetFile))
            restoredDf = pd.read_parquet(parquetFile)
        
        # Write to csv, in case we want to view it.
        bt.parquetToCsv(parquetFile)

    # -------------------------------------------------------------------------
    # testSamplesFile
    #
    # This is mostly for convenient viewing of a samples file in the 
    # debugger.
    # -------------------------------------------------------------------------
    def testSamplesFile(self):

        samples: np.ndarray = \
            np.fromfile(BuildTrainingTestCase._samplesFile, np.uint8). \
            reshape(Band.ROWS, Band.COLS)
            
        self.assertFalse((samples == 255).all())       
