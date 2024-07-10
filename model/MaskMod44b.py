import datetime
import logging
from pathlib import Path
import shutil

import numpy as np
from osgeo import gdal
from pyhdf import SD


# ----------------------------------------------------------------------------
# MaskMod44b
#
# The primary function of this class is to create copy a MOD44B HDF file
# from a specified directory and edit in place the copied HDF.
#
# Currently the edits are masking for 0% for subdatasets.
# ----------------------------------------------------------------------------
class MaskMod44b(object):

    MOD44B_PRE_STR: str = 'MOD44B.A'
    MASK_VALUE: int = 0
    REPLACEMENT_VALUE: int = 0
    MASK_DTYPE: np.dtype = np.int16

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, tile: str, year: int, hdfDir: Path,
                 maskDir: Path, subdataset: str, outDir: Path,
                 logger: logging.Logger) -> None:

        self._logger: logging.RootLogger = logger

        self._outputDir: Path = outDir
        self._outputDir.mkdir(exist_ok=True)

        self._subdataset: str = subdataset

        self._tile: str = tile

        self._year: int = year

        # Get the 0s mask
        self._maskDir: Path = maskDir
        maskSearchPattern = f'{self._tile}*0s.tif'
        self._maskPath: Path = self._getDatasetFilePath(self._maskDir,
                                                        maskSearchPattern)

        # Get the HDF file
        self._hdfDir: Path = hdfDir
        hdfSearchPattern = (
            f'{self.MOD44B_PRE_STR}{self._year}*.{self._tile}.*.hdf'
        )
        self._hdfSourcePath: Path = self._getDatasetFilePath(self._hdfDir,
                                                             hdfSearchPattern)
        # Editing HDF file will be inplace, make a copy to outDir
        self._hdfPath: Path = self._generateNewHdfFileName()
        # Copy source HDF to output path with new timestamp
        shutil.copyfile(self._hdfSourcePath, self._hdfPath)

    # ------------------------------------------------------------------------
    # _getDatasetFilePath
    #
    # Given a directory to search and a filename search pattern, glob for
    # files that match that path pattern. Raise any errors or warnings if no
    # or more than one files match the pattern. Return the path.
    # ------------------------------------------------------------------------
    def _getDatasetFilePath(self, directoryToSearch: Path,
                            searchPattern: str) -> Path:

        fullPathPattern = directoryToSearch / searchPattern

        # ---
        # To handle cases of multiple files matching pattern, sort the
        # filenames to get the first one in alpha-numerical order.
        # ---
        filesMatchingPattern = sorted(
            list(directoryToSearch.glob(searchPattern))
        )

        numberFilesMatching = len(filesMatchingPattern)

        # Handle case where no files match pattern.
        if numberFilesMatching == 0:

            errorMessage = (
                f'Found no files matching {fullPathPattern}'
            )

            raise RuntimeError(errorMessage)

        # Handle case where multiple files match pattern.
        elif numberFilesMatching > 1:

            warningMessage = (
                f'Found {numberFilesMatching} files matching'
                f' {fullPathPattern}. Using'
                f' first: {filesMatchingPattern[0]}'
            )

            self._logger.warning(warningMessage)

        # Take the 1st file matching pattern.
        filePath = filesMatchingPattern[0]

        return filePath

    # ------------------------------------------------------------------------
    # _generateNewHdfFileName
    #
    # Given a source HDF path, generate a new production timestamp and replace
    # the old with this new one. Append this new filename to the output dir.
    # This is our destination HDF path.
    # ------------------------------------------------------------------------
    def _generateNewHdfFileName(self) -> Path:

        # File name with no path.
        hdfSourceName = self._hdfSourcePath.name

        # Production time stamp refers to the time this hdf was produced.
        oldProductionTimeStamp = hdfSourceName.split('.')[4]

        self._logger.debug(
            f'Previous prod timestemp: {oldProductionTimeStamp}')

        # Generate a new production time stemp of YYYYDDDHHMMSS.
        newProductionTimeStamp = self._getProductionTimeStamp()

        self._logger.debug(f'New prod timestemp {newProductionTimeStamp}')

        # Replace the original timestemp with the newly generated one.
        newHdfFileName = hdfSourceName.replace(oldProductionTimeStamp,
                                               newProductionTimeStamp)

        # Join the new filename with the correct path, return.
        return self._outputDir / newHdfFileName

    # -------------------------------------------------------------------------
    # _getProductionTimeStamp
    #
    # Generate a timestamp of current time using format YYYYDDDHHMMSS
    # Y = year
    # D = julian day
    # H = hour
    # M = minute
    # S = second
    # -------------------------------------------------------------------------
    @staticmethod
    def _getProductionTimeStamp() -> str:
        sdtdate = datetime.datetime.now()
        year = sdtdate.year
        hm = sdtdate.strftime('%H%M%S')
        sdtdate = sdtdate.timetuple()
        jdate = sdtdate.tm_yday  # Julian day format
        post_str = '{}{:03}{}'.format(year, jdate, hm)
        return post_str

    # ------------------------------------------------------------------------
    # _readMaskToArray
    #
    # Read the mask dataset into a numpy array.
    # ------------------------------------------------------------------------
    def _readMaskToArray(self) -> np.ndarray:

        ds = gdal.Open(str(self._maskPath), gdal.GA_ReadOnly)

        maskArray = ds.ReadAsArray().astype(self.MASK_DTYPE)

        # Dereference to close
        ds = None

        return maskArray

    # ------------------------------------------------------------------------
    # run
    #
    # Given a mask and a destination HDF file, take the specified subdataset
    # array and mask. Write this masked array back to the subdataset within
    # the destination HDF.
    # ------------------------------------------------------------------------
    def run(self) -> None:

        self._logger.info(f'HDF source path: {self._hdfSourcePath}')
        self._logger.info(f'0s mask source path: {self._maskPath}')
        self._logger.info(f'HDF output path: {self._hdfPath}')
        self._logger.debug(f'HDF output exists? {self._hdfPath.exists()}')

        maskArray = self._readMaskToArray()
        self._logger.debug(maskArray)

        # HDF opening permission constants.
        writeKey = SD.SDC.WRITE
        createKey = SD.SDC.CREATE

        # Open HDF dataset.
        hdfScienceDataset = SD.SD(str(self._hdfPath), writeKey | createKey)

        # Select and read the specified subdataset.
        hdfSubdataset = hdfScienceDataset.select(self._subdataset)

        # Read the subdataset array into a numpy array.
        hdfSubdatasetArray = hdfSubdataset[:, :]

        self._logger.debug(hdfSubdatasetArray)
        self._logger.debug(
            f'HDF subdataset dtype: {hdfSubdatasetArray.dtype}'
        )

        # Mask where array is 0, make subdataset array 0.
        hdfSubdatasetArrayMasked = np.where(maskArray == self.MASK_VALUE,
                                            self.REPLACEMENT_VALUE,
                                            hdfSubdatasetArray)

        # Esnure we are matching the destination datasets data type.
        hdfSubdatasetArrayMasked = hdfSubdatasetArrayMasked.astype(
            hdfSubdatasetArray.dtype)

        self._logger.debug(hdfSubdatasetArrayMasked)
        self._logger.debug(
            f'HDF masked array dtype {hdfSubdatasetArrayMasked.dtype}'
        )

        # Write out masked array back to subdataset object.
        hdfSubdataset.set(hdfSubdatasetArrayMasked)

        # Close open file handlers
        hdfSubdataset.endaccess()
        hdfScienceDataset.end()
