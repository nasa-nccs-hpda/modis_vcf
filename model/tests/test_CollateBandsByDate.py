
import logging
from pathlib import Path
import sys
import tempfile
import unittest

from osgeo import gdal

from modis_vcf.model.CollateBandsByDate import CollateBandsByDate
from modis_vcf.model.Mate import Mate
from modis_vcf.model.Pair import Pair


# -----------------------------------------------------------------------------
# class CollateBandsByDateTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_CollateBandsByDate
# python -m unittest modis_vcf.model.tests.test_CollateBandsByDate.CollateBandsByDateTestCase.testRunOneBand
# -----------------------------------------------------------------------------
class CollateBandsByDateTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUpClass
    # -------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):

        # Define valid parameters.
        cls._validTileId = 'h09v05'
        cls._outDir = Path(tempfile.mkdtemp())
        cls._validYear = 2019
        
        cls._realInDir = \
            Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')

        cls._logger = logging.getLogger()
        cls._logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        cls._logger.addHandler(ch)

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        # Test invalid tile ID
        with self.assertRaisesRegex(RuntimeError, 'Invalid tile ID'):
            cbbd = CollateBandsByDate('y01v02',
                                      CollateBandsByDateTestCase._validYear,
                                      CollateBandsByDateTestCase._realInDir,
                                      CollateBandsByDateTestCase._outDir,
                                      CollateBandsByDateTestCase._logger)

        # Test invalid input directory.
        with self.assertRaisesRegex(RuntimeError, 'A valid input directory'):
            cbbd = CollateBandsByDate(CollateBandsByDateTestCase._validTileId,
                                      CollateBandsByDateTestCase._validYear,
                                      Path('/some/invalid/dir'),
                                      CollateBandsByDateTestCase._outDir,
                                      CollateBandsByDateTestCase._logger)

        # Test valid parameters.
        cbbd = CollateBandsByDate(CollateBandsByDateTestCase._validTileId,
                                  CollateBandsByDateTestCase._validYear,
                                  CollateBandsByDateTestCase._realInDir,
                                  CollateBandsByDateTestCase._outDir,
                                  CollateBandsByDateTestCase._logger)

    # -------------------------------------------------------------------------
    # testCollectInputFiles
    # -------------------------------------------------------------------------
    def testCollectInputFiles(self):
        
        cbbd = CollateBandsByDate(CollateBandsByDateTestCase._validTileId,
                                  CollateBandsByDateTestCase._validYear,
                                  CollateBandsByDateTestCase._realInDir,
                                  CollateBandsByDateTestCase._outDir,
                                  CollateBandsByDateTestCase._logger)

        self.assertEqual(23, len(cbbd.pairs))
        
        EXPECTED_MATES = [
            ('MOD44CH.A2019065.h09v05.061.2020290183523.hdf',
             'MOD44CQ.A2019065.h09v05.061.2020290183523.hdf'),
            ('MOD44CH.A2019081.h09v05.061.2020291181808.hdf',
             'MOD44CQ.A2019081.h09v05.061.2020291181808.hdf'),
            ('MOD44CH.A2019097.h09v05.061.2020292163055.hdf',
             'MOD44CQ.A2019097.h09v05.061.2020292163055.hdf'),
            ('MOD44CH.A2019113.h09v05.061.2020293185958.hdf',
             'MOD44CQ.A2019113.h09v05.061.2020293185958.hdf'),
            ('MOD44CH.A2019129.h09v05.061.2020294163306.hdf',  
             'MOD44CQ.A2019129.h09v05.061.2020294163306.hdf'),
            ('MOD44CH.A2019145.h09v05.061.2020298064017.hdf',  
             'MOD44CQ.A2019145.h09v05.061.2020298064017.hdf'),
            ('MOD44CH.A2019161.h09v05.061.2020298225029.hdf',  
             'MOD44CQ.A2019161.h09v05.061.2020298225029.hdf'),
            ('MOD44CH.A2019177.h09v05.061.2020303094028.hdf',
             'MOD44CQ.A2019177.h09v05.061.2020303094028.hdf'),
            ('MOD44CH.A2019193.h09v05.061.2020304043306.hdf',
             'MOD44CQ.A2019193.h09v05.061.2020304043306.hdf'),
            ('MOD44CH.A2019209.h09v05.061.2020304194354.hdf',
             'MOD44CQ.A2019209.h09v05.061.2020304194354.hdf'),
            ('MOD44CH.A2019225.h09v05.061.2020306160356.hdf',
             'MOD44CQ.A2019225.h09v05.061.2020306160355.hdf'),
            ('MOD44CH.A2019241.h09v05.061.2020308202239.hdf',
             'MOD44CQ.A2019241.h09v05.061.2020308202239.hdf'),
            ('MOD44CH.A2019257.h09v05.061.2020312061813.hdf',  
             'MOD44CQ.A2019257.h09v05.061.2020312061813.hdf'),
            ('MOD44CH.A2019273.h09v05.061.2020314024311.hdf',
             'MOD44CQ.A2019273.h09v05.061.2020314024310.hdf'),
            ('MOD44CH.A2019289.h09v05.061.2020316003828.hdf',
             'MOD44CQ.A2019289.h09v05.061.2020316003828.hdf'),
            ('MOD44CH.A2019305.h09v05.061.2020317205306.hdf',  
             'MOD44CQ.A2019305.h09v05.061.2020317205303.hdf'),
            ('MOD44CH.A2019321.h09v05.061.2020319040840.hdf',
             'MOD44CQ.A2019321.h09v05.061.2020319040840.hdf'),
            ('MOD44CH.A2019337.h09v05.061.2020321233219.hdf',
             'MOD44CQ.A2019337.h09v05.061.2020321233219.hdf'),
            ('MOD44CH.A2019353.h09v05.061.2020323091840.hdf',
             'MOD44CQ.A2019353.h09v05.061.2020323091840.hdf'),
            ('MOD44CH.A2020001.h09v05.061.2020326040333.hdf',
             'MOD44CQ.A2020001.h09v05.061.2020326040333.hdf'),
            ('MOD44CH.A2020017.h09v05.061.2020328175533.hdf',
             'MOD44CQ.A2020017.h09v05.061.2020328175533.hdf'),
            ('MOD44CH.A2020033.h09v05.061.2020329104120.hdf',
             'MOD44CQ.A2020033.h09v05.061.2020329104120.hdf'),
            ('MOD44CH.A2020049.h09v05.061.2020335041038.hdf',
             'MOD44CQ.A2020049.h09v05.061.2020335041037.hdf')
        ]
        
        self.assertEqual(len(cbbd.pairs), len(EXPECTED_MATES))
            
        for pair in EXPECTED_MATES:
            
            key = Mate.getKey(Path(pair[0]))
            cbbdPair = cbbd._pairs[key]
            self.assertEqual(cbbdPair.chMate.fileName.name, pair[0])
            self.assertEqual(cbbdPair.cqMate.fileName.name, pair[1])

    # -------------------------------------------------------------------------
    # testParseTileId
    # -------------------------------------------------------------------------
    def testParseTileId(self):

        CollateBandsByDate.parseTileId('h09v05')

        with self.assertRaisesRegex(RuntimeError, 'Invalid tile ID'):
            CollateBandsByDate.parseTileId('h007v05')
        
    # -------------------------------------------------------------------------
    # testRunOneBand
    # -------------------------------------------------------------------------
    def testRunOneBand(self):

        cbbd = CollateBandsByDate(CollateBandsByDateTestCase._validTileId,
                                  CollateBandsByDateTestCase._validYear,
                                  CollateBandsByDateTestCase._realInDir,
                                  CollateBandsByDateTestCase._outDir,
                                  CollateBandsByDateTestCase._logger)

        # Test an invalid band name.
        with self.assertRaisesRegex(RuntimeError, 
                                    'Unable to determine mate for band' +
                                    ' MilliVanilli'):

            cbbd.runOneBand('MilliVanilli')
        
        # Test a valid band.
        band = cbbd.runOneBand(Pair.BAND5)
        self.assertEqual(len(band.dayXref), 23)

        # If the same band is run, it should not need to process again.
        print('Running same band again.  There should be no processing.')
        band = cbbd.runOneBand(Pair.BAND5)

    # -------------------------------------------------------------------------
    # testGetBand
    # -------------------------------------------------------------------------
    def testGetBand(self):

        cbbd = CollateBandsByDate(CollateBandsByDateTestCase._validTileId,
                                  CollateBandsByDateTestCase._validYear,
                                  CollateBandsByDateTestCase._realInDir,
                                  CollateBandsByDateTestCase._outDir,
                                  CollateBandsByDateTestCase._logger)

        band = cbbd.getBand(Pair.BAND1)
        self.assertEqual(len(band.dayXref), 23)

    # -------------------------------------------------------------------------
    # testPairs
    # -------------------------------------------------------------------------
    def testPairs(self):
        
        cbbd = CollateBandsByDate(CollateBandsByDateTestCase._validTileId,
                                  CollateBandsByDateTestCase._validYear,
                                  CollateBandsByDateTestCase._realInDir,
                                  CollateBandsByDateTestCase._outDir,
                                  CollateBandsByDateTestCase._logger)

        pairs = cbbd.pairs
        self.assertEqual(len(pairs), 23)
        