import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from modis_vcf.model.MaskMod44b import MaskMod44b


# ----------------------------------------------------------------------------
# class MaskMod44bTestCase
#
# python -m unittest discover modis_vcf/model/tests
# python -m unittest modis_vcf.model.tests.test_MaskMod44b
# ----------------------------------------------------------------------------
class MaskMod44bTestCase(unittest.TestCase):

    # ------------------------------------------------------------------------
    # setUpClass
    # ------------------------------------------------------------------------
    @classmethod
    def setUpClass(self):

        self._tile0 = 'h10v03'
        self._tile1 = 'h09v01'
        self._year = 2019
        self._subdataset = 'Tree'
        self._outDir = Path(tempfile.gettempdir())
        self._tempDir = Path(tempfile.gettempdir())

        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        self._logger.addHandler(ch)

    # ------------------------------------------------------------------------
    # testInitNoArgs
    # ------------------------------------------------------------------------
    def testInitNoArgs(self):

        with self.assertRaises(TypeError):
            MaskMod44b()

    # ------------------------------------------------------------------------
    # testInitSingleFiles
    # ------------------------------------------------------------------------
    def testInitSingleFiles(self):

        hdfPrefix = (
            f'{MaskMod44b.MOD44B_PRE_STR}{self._year}001.{self._tile0}.00'
        )
        _, tempHdf = tempfile.mkstemp(suffix='.hdf',
                                      prefix=hdfPrefix)

        _, tempMask = tempfile.mkstemp(suffix='0s.tif',
                                       prefix=self._tile0)

        self._logger.info('Temp files')
        self._logger.info(tempHdf)
        self._logger.info(tempMask)

        # Test normal init
        _ = MaskMod44b(tile=self._tile0,
                       year=self._year,
                       hdfDir=self._tempDir,
                       maskDir=self._tempDir,
                       outDir=self._outDir,
                       subdataset=self._subdataset,
                       logger=self._logger)

        os.remove(tempHdf)
        os.remove(tempMask)

    # ------------------------------------------------------------------------
    # testInitMissingMask
    # ------------------------------------------------------------------------
    def testInitMissingMask(self):

        hdfPrefix = (
            f'{MaskMod44b.MOD44B_PRE_STR}{self._year}001.'
            f'{self._tile1}.00'
        )
        _, tempHdf = tempfile.mkstemp(suffix='.hdf',
                                      prefix=hdfPrefix)
        self._logger.info(tempHdf)

        with self.assertRaisesRegex(RuntimeError, 'Found no files'):

            # Test normal init
            _ = MaskMod44b(tile=self._tile1,
                           year=self._year,
                           hdfDir=self._tempDir,
                           maskDir=self._tempDir,
                           outDir=self._outDir,
                           subdataset=self._subdataset,
                           logger=self._logger)

        os.remove(tempHdf)

    # ------------------------------------------------------------------------
    # testInitMissingHdf
    # ------------------------------------------------------------------------
    def testInitMissingHdf(self):

        _, tempMask = tempfile.mkstemp(suffix='0s.tif',
                                       prefix=self._tile1)

        self._logger.info(tempMask)

        with self.assertRaisesRegex(RuntimeError, 'Found no files'):

            # Test normal init
            _ = MaskMod44b(tile=self._tile1,
                           year=self._year,
                           hdfDir=self._tempDir,
                           maskDir=self._tempDir,
                           outDir=self._outDir,
                           subdataset=self._subdataset,
                           logger=self._logger)

        os.remove(tempMask)

    # ------------------------------------------------------------------------
    # testGenerateNewHdfFileName
    # ------------------------------------------------------------------------
    @patch.object(MaskMod44b, '_getProductionTimeStamp')
    def testGenerateNewHdfFileName(self, mock_getProductionTimeStemp):
        mock_getProductionTimeStemp.return_value = '2024191153045'

        hdfPrefix = (
            f'{MaskMod44b.MOD44B_PRE_STR}{self._year}.001.'
            f'{self._tile0}.2023150123456.'
        )
        _, tempHdf = tempfile.mkstemp(suffix='.hdf',
                                      prefix=hdfPrefix)

        _, tempMask = tempfile.mkstemp(suffix='0s.tif',
                                       prefix=self._tile0)

        self._logger.info('Temp files')
        self._logger.info(tempHdf)
        self._logger.info(tempMask)

        # Test normal init
        maskMod44b = MaskMod44b(tile=self._tile0,
                                year=self._year,
                                hdfDir=self._tempDir,
                                maskDir=self._tempDir,
                                outDir=self._outDir,
                                subdataset=self._subdataset,
                                logger=self._logger)

        result = str(maskMod44b._generateNewHdfFileName())

        expectedNewFileName = tempHdf.replace('2023150123456', '2024191153045')
        self._logger.info(result)
        self._logger.info(expectedNewFileName)
        self.assertEqual(result, str(expectedNewFileName))

        os.remove(tempHdf)
        os.remove(tempMask)
