
import logging
import numpy as np
from pathlib import Path

from osgeo import gdal

from modis_vcf.model.Band import Band
from modis_vcf.model.Mate import Mate


# ----------------------------------------------------------------------------
# Class Pair
#
# 1MOD44CH:Band_3           Int16
# 2MOD44CH:Band_4           Int16
# 3MOD44CH:Band_5           Int16
# 4MOD44CH:Band_6           Int16
# 5MOD44CH:Band_7           Int16
# 6MOD44CH:QA_500m
# 7MOD44CH:"water flags"
# 8MOD44CH:"land flags"
# 9MOD44CH:Band_20
# 10MOD44CH:Band_31         UInt16
# 11MOD44CH:Band_32
# 12MOD44CH:"total flags"
#
# 1MOD44CQ:state            UInt16  GDT_Byte=1  uint8
# 2MOD44CQ:view_angle
# 3MOD44CQ:solar_zenith     Byte
# 4MOD44CQ:azimuth
# 5MOD44CQ:Band_1           Int16   GDT_Int16=3 int16
# 6MOD44CQ:Band_2           Int16
# 7MOD44CQ:QA_250m          UInt16  GDT_Byte=1  uint8
# 8MOD44CQ:obscov
# 9MOD44CQ:orbit_pnt
#
# Note: GDT_Byte is the same as an uint8, per
# https://naturalatlas.github.io/node-gdal/classes/Constants%20(GDT).html#prop-gdal.GDT_Byte
#
# TODO: write accessors
# ----------------------------------------------------------------------------
class Pair(object):

    BAND1 = 'Band_1'
    BAND2 = 'Band_2'
    BAND3 = 'Band_3'
    BAND4 = 'Band_4'
    BAND5 = 'Band_5'
    BAND6 = 'Band_6'
    BAND7 = 'Band_7'
    BAND31 = 'Band_31'
    BANDS = [BAND1, BAND2, BAND3, BAND4, BAND5, BAND6, BAND7]

    STATE_INDEX = 0
    SOLAR_ZENITH_INDEX = 3
    ZENITH_CUTOFF = 72

    # ---
    # This is a mapping from band names, as seen in the HDF files, to
    # their index in the HDF files.
    # ---
    CH_DS_XREF = {BAND3: 0, BAND4: 1, BAND5: 2, BAND6: 3, BAND7: 4, 
                  BAND31: 9}
                  
    CQ_DS_XREF = {BAND1: 4, BAND2: 5}

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 chMate: Mate, 
                 cqMate: Mate, 
                 logger: logging.RootLogger=None):
        
        self._chMate: Mate = chMate
        self._cqMate: Mate = cqMate
        self._qaMask: np.ndarray = None
        self._solarZenith: np.ndarray = None
        self._logger: logging.RootLogger = logger
        
    # ------------------------------------------------------------------------
    # createQaMask
    # ------------------------------------------------------------------------
    def _createQaMask(self) -> np.ndarray:
        
        stateName = self._cqMate.dataset.GetSubDatasets()[Pair.STATE_INDEX][0]
        state = gdal.Open(stateName).ReadAsArray().astype(np.int16)
        cloud = state & 3
        shadow = state & 4
        adjacency = state & 8192
        aerosol = (state & 192) >> 6

        mask = np.where((cloud == 0) &
                        (shadow == 0) &
                        (aerosol != 3) &   
                        (adjacency == 0) & 
                        (self.solarZenith < Pair.ZENITH_CUTOFF),
                        1,
                        Band.NO_DATA).astype(np.int16)

        return mask
        
    # -------------------------------------------------------------------------
    # chMate
    # -------------------------------------------------------------------------
    @property
    def chMate(self) -> Mate:
        return self._chMate
        
    # -------------------------------------------------------------------------
    # cqMate
    # -------------------------------------------------------------------------
    @property
    def cqMate(self) -> Mate:
        return self._cqMate
        
    # -------------------------------------------------------------------------
    # qaMask
    # -------------------------------------------------------------------------
    @property
    def qaMask(self) -> np.ndarray:

        # ---
        # Once self._qaMask is assigned a Numpy array, self_qaMask == None 
        # causes a ValueError about "The truth value of an array with more 
        # than one element is ambiguous."  Trap that error.
        # ---
        try:
        
            if not self._qaMask:
                self._qaMask = self._createQaMask()

        except ValueError:
            pass
            
        return self._qaMask

    # ------------------------------------------------------------------------
    # getMate
    # ------------------------------------------------------------------------
    def _getMate(self, bandName: str) -> (Mate, int):
        
        mate = None
        index = -1
        
        if bandName in Pair.CH_DS_XREF:

            mate = self._chMate
            index = Pair.CH_DS_XREF[bandName]
            
        elif bandName in Pair.CQ_DS_XREF:
            
            mate = self._cqMate
            index = Pair.CQ_DS_XREF[bandName]
            
        else:
            raise RuntimeError('Unable to determine mate for band ' + 
                               bandName)
                 
        return mate, index
                      
    # ------------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------------
    def read(self, bandName: str, applyQa: bool = False) -> (np.ndarray, int):
        
        # Read the raw band from the correct mate.
        mate, index = self._getMate(bandName)
        rawBand, dataType = mate.read(index)  # 'int16'
        
        # ---
        # If a pixel value is larger than the solar zenith cut off, clamp it
        # to 16000. Outband.dtype = 'int16'.
        # ---
        outBand = np.where(self.solarZenith < Pair.ZENITH_CUTOFF,
                           rawBand,
                           16000)
        
        # Do not apply QA to solar zenith.
        if applyQa:
            outBand = np.where(self.qaMask==1, outBand, Band.NO_DATA)
            
        return outBand, dataType
        
    # ------------------------------------------------------------------------
    # getSolarZ
    # ------------------------------------------------------------------------
    @property
    def solarZenith(self) -> np.ndarray:

        try:

            if not self._solarZenith:

                self._solarZenith = self._cqMate.read( \
                    Pair.SOLAR_ZENITH_INDEX)[0]

        except ValueError:
            pass

        return self._solarZenith
        
    # -------------------------------------------------------------------------
    # writeMask
    # -------------------------------------------------------------------------
    def writeMask(self, outDir: Path) -> None:

        outName = outDir / 'mask.tif'
        
        if self._logger:
            self._logger.info('Writing ' + str(outName))

        dataType = gdal.GDT_UInt16

        ds = gdal.GetDriverByName('GTiff').Create(
            str(outName),
            Band.ROWS,
            Band.COLS,
            1,
            dataType,
            options=['COMPRESS=LZW', 'BIGTIFF=YES'])

        ds.SetSpatialRef(Band.modisSinusoidal)

        gdBand = ds.GetRasterBand(1)
        gdBand.WriteArray(self._qaMask)
        gdBand.SetMetadata({'Name': 'QA Mask'})
        gdBand.FlushCache()
        gdBand = None
        ds = None
        