
from enum import Enum
import logging
from pathlib import Path

import numpy as np

from modis_vcf.model.Band import Band
from modis_vcf.model.Mate import Mate
from modis_vcf.model.Pair import Pair


# ----------------------------------------------------------------------------
# Class CollateBandsByDate
#
# TODO: Add NDVI to the stack or do it in MakeMetrics?
# ----------------------------------------------------------------------------
class CollateBandsByDate(object):
    
    START_DAY = 65
    FileType = Enum('FileType', ['CH', 'CQ'])
    
    # ------------------------------------------------------------------------
    # __init__
    #
    # The application of QA must be set for the instance.  Otherwise, calls
    # to runOneBand could populate self._bands with some bands that have QA 
    # and some bands that do not.
    # ------------------------------------------------------------------------
    def __init__(self, 
                 tileId: str, 
                 year: int, 
                 inputDir: Path, 
                 outputDir: Path,
                 logger: logging.RootLogger,
                 applyQa: bool = True,
                 write: bool = False):
        
        if not year:
            raise RuntimeException('A year must be provided.')
            
        # Tile ID
        CollateBandsByDate.parseTileId(tileId)
        self._tileId = tileId
        
        # Let an invalid year fail during the search.
        self._year = year
        
        # Input directory
        if not inputDir or not inputDir.exists() or not inputDir.is_dir():
            raise RuntimeError('A valid input directory must be provided.')
            
        self._inputDir: Path = inputDir

        # Output directory
        if not outputDir or not outputDir.exists() or not outputDir.is_dir():
           
            raise RuntimeError('A valid output directory must be provided.')
            
        self._outputDir: Path = outputDir

        self._applyQa = applyQa
        self._write = write

        # Logger
        if not logger:
            raise RuntimeError('A logger must be provided.')
            
        self._logger: logger = logger

        # ---
        # Pairs {yyyydd: Pair}
        # ---
        self._pairs = {}
        
        # ---
        # Bands
        # {Band: {Band Name: Band}
        # ---
        self._bands = {}
        
    # ------------------------------------------------------------------------
    # collectInputFiles
    # ------------------------------------------------------------------------
    def _collectInputFiles(self) -> None:

        if self._logger:
            self._logger.info('Collecting input files for ' + self._tileId)

        filesInYear = 23
        step = 16

        for i in range(0, filesInYear):
            
            sequentialDay = CollateBandsByDate.START_DAY + i * step
            julianDay = sequentialDay % 368  # to wrap to next year
            year = self._year if sequentialDay < 365 else self._year + 1
            yearAndJulian = str(year) + str(julianDay).zfill(3)
            
            chGlob = 'MOD44CH.A' + \
                     yearAndJulian + \
                     '.' + \
                     self._tileId + \
                     '.*.hdf'
            
            chFiles = list(self._inputDir.glob(chGlob))
            
            if not chFiles:
                
                msg = 'No CH files found for ' + yearAndJulian
                # raise RuntimeError(msg)
                self._logger.warn(msg)
                continue

            chFile = chFiles[0]
            
            if not chFile.exists():
                raise RuntimeError('Unable to find ' + str(chFile))
                
            cqGlob = chGlob.replace('CH', 'CQ')
            cqFile = list(self._inputDir.glob(cqGlob))[0]
        
            if not cqFile.exists():
                raise RuntimeError('Unable to find ' + str(cqFile))
                
            # Store in the pairs dictionary.
            chMate = Mate(chFile)
            cqMate = Mate(cqFile)
            key = chMate.getMyKey()
            
            if key not in self._pairs:
                self._pairs[key] = Pair(chMate, cqMate)
        
    # ------------------------------------------------------------------------
    # getBand
    # ------------------------------------------------------------------------
    def getBand(self, bandName: str) -> Band:
        
        if bandName not in self._bands:

            bandFileName = self._outputDir / (bandName + '.tif')

            if bandFileName.exists():

                band = Band()
                band.read(bandFileName)
                self._bands[bandName] = band

            else:

                band = self.runOneBand(bandName)
                
                if self._write:
                    band.write(self._outputDir)
                
        return self._bands[bandName]
        
    # -------------------------------------------------------------------------
    # logger
    # -------------------------------------------------------------------------
    @property
    def logger(self) -> logging.RootLogger:
        
        return self._logger
        
    # -------------------------------------------------------------------------
    # pairs
    # -------------------------------------------------------------------------
    @property
    def pairs(self) -> dict:

        if not self._pairs:
            self._collectInputFiles()
            
        return self._pairs

    # ------------------------------------------------------------------------
    # parseTileId
    # ------------------------------------------------------------------------
    @staticmethod
    def parseTileId(tileId: str) -> list:
        
        if tileId[0].lower() != 'h' or tileId[3] != 'v':
            raise RuntimeError('Invalid tile ID, ' + tileId)
            
        return int(tileId[1:3]), int(tileId[4:6])
        
    # ------------------------------------------------------------------------
    # runOneBand
    # ------------------------------------------------------------------------
    def runOneBand(self, bandName: str) -> Band:

        if bandName in self._bands:
            return self._bands[bandName]
            
        if self._logger:
            self._logger.info('Processing ' + bandName)
            
        dataType: int = None
        dayIndex = 0
        dayXref = {}  # yyyydd: index into cube

        cube = np.full((len(self.pairs), Band.ROWS, Band.COLS), 
                       np.nan,
                       dtype=Band.NUMPY_DTYPE)
        
        for key in self.pairs:
            
            if self._logger:
                self._logger.info('Processing ' + key)
                
            pair: Pair = self.pairs[key]
            day, dataType = pair.read(bandName, self._applyQa)
            dayXref[key] = dayIndex
            cube[dayIndex] = day
            dayIndex += 1
            
        # ---
        # Create the band.
        # ---
        band = Band(bandName, cube, dayXref, self._logger)
        self._bands[bandName] = band
        
        return band
                    