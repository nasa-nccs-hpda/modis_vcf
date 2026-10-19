
from concurrent.futures import ThreadPoolExecutor
import joblib
import logging
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
# CPU/sklearn: models are now BaggingRegressor (see
# Sync_Plan_and_Model_Retraining_Alternate.md), not RandomForest.
# Aliased so the RandomForestRegressor type hints below still read
# correctly without renaming each one.
from sklearn.ensemble import BaggingRegressor as RandomForestRegressor
# GPU/cuML (forest203 has no GPU; not currently usable):
# from cuml.ensemble import RandomForestRegressor

from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr

import json

from modis_vcf.model import BlendA
from modis_vcf.model.BandDayFile import BandDayFile
from modis_vcf.model.CompactForest import CompactForest
from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
from modis_vcf.model.ProductTypeMod44W import ProductTypeMod44W
 
MOD44_DIR = Path('/css/modis/Collection6.1/L3/MOD44B-VCF/dev')
MOD44W_DIR = Path('/css/modis/Collection6.1/L3/MOD44W-LandWaterMask')


# ----------------------------------------------------------------------------
# VcfPredict
# ----------------------------------------------------------------------------
class VcfPredict(object):
    
    MASK_WATER = 1
    MASK_OUT_OF_PROJ = 250
    PRED_WATER = 200
    PRED_OUT_OF_PROJ = 250
    PREDICT_CHUNK_ROWS = 1000000

    # Output storage, as in the historical MOD44B: the three percent layers are uint8 (0-100, water
    # 200, out of projection 250, fill 253) and the SD layers are uint16 holding SD x 100.
    PCT_DATA_TYPE = gdal.GDT_Byte
    PCT_NO_DATA = 253
    SD_DATA_TYPE = gdal.GDT_UInt16

    # The MODIS sinusoidal grid: a sphere of radius 6371007.181 m, 36 x 18 tiles of 1111950.5197665 m
    # (4800 x 4800 pixels of 231.656 m at 250 m), upper left corner of h00v00 at (-20015109.354, 10007554.677).
    MODIS_SINU = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs'
    MODIS_TILE_SIZE = 1111950.5196666666
    MODIS_GRID_ULX = -20015109.354
    MODIS_GRID_ULY = 10007554.677

    # How the ensemble members' predictions are combined into the value of a pixel.
    # 'mean' is what predict() returns; 'mid20' drops the MID20_TRIM highest and lowest members
    # (the middle 20 of 30) and averages the rest; 'median' is the members' median.
    AGG_METHODS = ('mean', 'mid20', 'median')
    MID20_TRIM = 5

    # Tile rows (the v of hXXvYY) where land is set to tree cover 0, non-vegetated 100.
    HIGH_LATITUDE_BARE_VTILES = (0,)

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self,
                 treeCoverModelFile: Path,
                 treeCoverTopNFile: Path,
                 nonvegModelFile: Path,
                 nonvegModelTopNFile: Path,
                 outDir: Path,
                 metricsDir: Path,
                 mod44Dir: Path = MOD44_DIR,
                 mod44wDir: Path = MOD44W_DIR,
                 logger: logging.RootLogger = None,
                 nJobs: int = 1,
                 treeCoverAgg: str = 'mean',
                 nonvegAgg: str = 'mean',
                 writeSd: bool = True,
                 partnerModelFile: Path = None,
                 partnerTopNFile: Path = None,
                 partnerMetricsDir: Path = None,
                 countsDir: Path = None,
                 partnerAgg: str = 'median',
                 bp10: bool = True):

        # Validate the output directory.
        if outDir is None or not outDir.exists() or not outDir.is_dir():
            raise RuntimeError('A valid output directory must be provided.')
            
        self._outDir: Path = outDir
        
        # Load the models.
        self._treeCoverRf: RandomForestRegressor = None
        self._treeCoverTopN: list = None
        
        self._treeCoverRf: RandomForestRegressor = VcfPredict._loadModel(treeCoverModelFile)
            
        with open(treeCoverTopNFile, 'rb') as f:
            self._treeCoverTopN = pickle.load(f)
            
        self._nonvegRf: RandomForestRegressor = None
        self._nonvegTopN: list = None
        
        self._nonvegRf: RandomForestRegressor = VcfPredict._loadModel(nonvegModelFile)
            
        with open(nonvegModelTopNFile, 'rb') as f:
            self._nonvegTopN = pickle.load(f)
            
        # Threads used by the models when predicting.
        self._nJobs: int = nJobs
        self._treeCoverRf.set_params(n_jobs=nJobs)
        self._nonvegRf.set_params(n_jobs=nJobs)

        # How each layer's members are combined, and whether to write the SD layers.
        for model, agg in ((self._treeCoverRf, treeCoverAgg), (self._nonvegRf, nonvegAgg)):

            if agg not in VcfPredict.AGG_METHODS:
                raise ValueError('Unknown aggregation ' + str(agg) + '; use one of ' + str(VcfPredict.AGG_METHODS))

            if agg != 'mean' and VcfPredict._members(model) is None:
                raise ValueError(type(model).__name__ + ' has no ensemble members, so it only supports ' +
                                 'aggregation "mean".')

        self._treeCoverAgg: str = treeCoverAgg
        self._nonvegAgg: str = nonvegAgg
        self._writeSd: bool = writeSd

        # ---
        # Blend A partner (BlendA): a tree model on the V4r metrics, blended
        # into the tree layer on rows v06-v11 with the cloud counts in
        # countsDir.  Without a partner the tree layer is the clear model's.
        # ---
        self._partnerRf = None
        self._partnerTopN: list = None
        self._partnerMetricsDir: Path = partnerMetricsDir
        self._countsDir: Path = countsDir
        self._partnerAgg: str = partnerAgg

        if partnerModelFile is not None:

            if partnerTopNFile is None or partnerMetricsDir is None or countsDir is None:
                raise ValueError('Blend A needs the partner model, its top-N file, the partner (V4r) metrics '
                                 'directory and the cloud counts directory.')

            self._partnerRf = VcfPredict._loadModel(partnerModelFile)
            self._partnerRf.set_params(n_jobs=nJobs)

            with open(partnerTopNFile, 'rb') as f:
                self._partnerTopN = pickle.load(f)

        # BP10 (BlendA.bp10), the last step of the product since v1.4.0.
        self._bp10: bool = bp10

        # Instantiate the product type for the metrics.
        self._productType = ProductTypeMod44(mod44Dir)
        self._metricsDir: Path = metricsDir
        
        # Instantiate a objects for the water masks.
        self._productTypeM44W = ProductTypeMod44W(mod44wDir)
        self._waterMasks = {}  # {tidyear: water mask}
        
        # Instantiate the logger.
        if not logger:
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            if (not logger.hasHandlers()):

                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                logger.addHandler(ch)

        self._logger: logging.RootLogger = logger

    # ------------------------------------------------------------------------
    # getMask
    # ------------------------------------------------------------------------
    def _getMask(self, tid: str, year: int) -> np.ndarray:
        
        key = tid + str(year)
        
        if key not in self._waterMasks:
            
            bdf = BandDayFile(). \
                initFromParams(self._productTypeM44W,
                               self._productTypeM44W.WATER_MASK,
                               tid,
                               year,
                               1,
                               self._outDir,
                               self._logger)
        
            band: np.ndarray = bdf.raster(applyQa=False)
            self._waterMasks[key] = band
        
        return self._waterMasks[key]

    # ------------------------------------------------------------------------
    # getMetrics
    # ------------------------------------------------------------------------
    def _getMetrics(self, tid: str, year: int, rfNames: list, metricsDir: Path = None,
                    lowSun: str = None) -> pd.DataFrame:

        names = rfNames
        
        self._logger.info('Retrieving metrics: ' + str(names))
        
        mInstance = Metrics(tid, 
                            year, 
                            self._productType, 
                            metricsDir or self._metricsDir, 
                            self._logger,
                            lowSun=lowSun)
        
        metrics = pd.DataFrame()
        
        for i in range(len(names)):
            metrics[names[i]] = mInstance.getMetricFromRf(names[i]).ravel()

        return metrics
        
    # ------------------------------------------------------------------------
    # getTable
    #
    # The predictors of rfNames as one float32 table (pixels x distinct names),
    # read in blocks (Metrics.getTableFromRf).  Returns (table, column names).
    # ------------------------------------------------------------------------
    def _getTable(self, tid: str, year: int, rfNames: list, metricsDir: Path = None, lowSun: str = None) -> tuple:

        self._logger.info('Retrieving metrics (blocks): ' + str(rfNames))
        mInstance = Metrics(tid, year, self._productType, metricsDir or self._metricsDir, self._logger, lowSun=lowSun)
        return mInstance.getTableFromRf(rfNames, dtype=np.float32)

    # ------------------------------------------------------------------------
    # maskPrediction
    # ------------------------------------------------------------------------
    def _maskPrediction(self, 
                        tid: str, 
                        year: int, 
                        prediction: np.ndarray) -> np.ndarray:
        
        mask: np.ndarray = self._getMask(tid, year)
        
        maskedPred = np.where(mask == VcfPredict.MASK_WATER,
                              VcfPredict.PRED_WATER, 
                              prediction)
                              
        maskedPred = np.where(mask == VcfPredict.MASK_OUT_OF_PROJ,
                              VcfPredict.PRED_OUT_OF_PROJ, 
                              maskedPred)
        
        return maskedPred
        
    # ------------------------------------------------------------------------
    # runPrediction
    # ------------------------------------------------------------------------
    def _runPrediction(self,
                       tid: str,
                       year: int,
                       rf: RandomForestRegressor,
                       topN: list,
                       agg: str = 'mean',
                       metricsDir: Path = None,
                       lowSun: str = None) -> tuple:
        """
        Returns (prediction, sd).  prediction is the masked int16 layer; sd is the members' population SD stored
        as round(100 * SD) (0 on water and out-of-projection), or None when the SD is switched off or the model
        has no ensemble members (e.g. XGBoost).
        """
        # ---
        # Ensemble models: one pass over the members gives the value (combined by agg) and the SD.
        # The members' mean equals predict() exactly for Random Forest and BaggingRegressor
        # (Reports/SD_Layer_Results.md), so 'mean' gives the same layer as the predict() path below.
        # The float32 table is read straight from the metric files in blocks (Metrics.getTableFromRf):
        # the same values as _getMetrics(...).to_numpy(dtype=np.float32), without whole files or a second copy.
        # ---
        if VcfPredict._members(rf) is not None and (self._writeSd or agg != 'mean'):

            X, _ = self._getTable(tid, year, topN, metricsDir, lowSun)

            self._logger.info('Predicting with the ensemble members (' + agg + ') ...')

            value, sd = VcfPredict._memberPredict(rf, X, agg, self._nJobs)
            del X

            prediction = np.clip(VcfPredict._roundValue(value).reshape(self._productType.ROWS,
                                                                       self._productType.COLS), 0, 100)

            maskedPred: np.ndarray = self._maskPrediction(tid, year, prediction)

            sdRaster = None

            if self._writeSd:

                sdRaster = np.round(sd * 100).astype(np.int16).reshape(self._productType.ROWS,
                                                                       self._productType.COLS)
                mask = self._getMask(tid, year)
                outside = (mask == VcfPredict.MASK_WATER) | (mask == VcfPredict.MASK_OUT_OF_PROJ)
                sdRaster = np.where(outside, 0, sdRaster).astype(np.int16)

            return maskedPred, sdRaster

        X: pd.DataFrame = self._getMetrics(tid, year, topN, metricsDir, lowSun)

        if self._writeSd:
            self._logger.info(type(rf).__name__ + ' has no ensemble members: SD layer skipped.')
        # X = X.replace(-10001, 10001)
        
        self._logger.info('Predicting ...')
        
        # GPU/cuML: .predict() returns a cuDF object needing .to_numpy().
        # Not currently usable (forest203 has no GPU).
        # prediction: np.ndarray = rf.predict(X). \
        #                          to_numpy(). \
        #                          astype(np.int16). \
        #                          reshape(self._productType.ROWS,
        #                                  self._productType.COLS)

        # CPU/sklearn: .predict() already returns a plain numpy ndarray.
        # Predict in blocks of rows with threads, which bounds memory and uses every core.
        with joblib.parallel_backend('threading', n_jobs=self._nJobs):
            parts = [rf.predict(X.iloc[i:i + VcfPredict.PREDICT_CHUNK_ROWS])
                     for i in range(0, len(X), VcfPredict.PREDICT_CHUNK_ROWS)]

        prediction: np.ndarray = VcfPredict._roundValue(np.concatenate(parts)). \
                                 reshape(self._productType.ROWS,
                                         self._productType.COLS)
        del X, parts

        # Clamp to the valid percentage range before applying the water/out-of-projection
        # codes below (200 and 250 are legitimate values outside 0-100 and must not be
        # clamped).  Random Forest/Bagging predictions are already bounded to the training
        # range by construction (an average of decision-tree leaf means), so this never
        # changes their output; boosted models such as XGBoost are not bounded this way and
        # can genuinely predict outside 0-100.
        prediction = np.clip(prediction, 0, 100)

        # Apply the water mask.
        maskedPred: np.ndarray = self._maskPrediction(tid, year, prediction)

        return maskedPred, None

    # ------------------------------------------------------------------------
    # vTile
    # ------------------------------------------------------------------------
    @staticmethod
    def _hTile(tid: str) -> int:
        """The tile column of a tile ID such as 'h16v00'."""
        return int(tid[tid.index('h') + 1:tid.index('v')])

    # ------------------------------------------------------------------------
    # geoTransform
    # ------------------------------------------------------------------------
    @staticmethod
    def _geoTransform(tid: str, shape: tuple) -> tuple:
        """The GDAL geotransform of a tile of the MODIS sinusoidal grid with shape (rows, columns)."""
        size = VcfPredict.MODIS_TILE_SIZE
        ulx = VcfPredict.MODIS_GRID_ULX + VcfPredict._hTile(tid) * size
        uly = VcfPredict.MODIS_GRID_ULY - VcfPredict._vTile(tid) * size
        return (ulx, size / shape[1], 0.0, uly, 0.0, -size / shape[0])

    # ------------------------------------------------------------------------
    # vTile
    # ------------------------------------------------------------------------
    @staticmethod
    def _vTile(tid: str) -> int:
        """The tile row of a tile ID such as 'h16v00'."""
        return int(tid[tid.index('v') + 1:])

    # ------------------------------------------------------------------------
    # roundValue
    # ------------------------------------------------------------------------
    @staticmethod
    def _roundValue(value: np.ndarray) -> np.ndarray:
        """
        Round to the nearest integer, halves up, as the C code does (floor(prediction + 0.5), MOD44B3.c).
        Truncating instead biased both layers down by about half a point
        (Reports/Rounding_vs_Truncation_Results.md).
        """
        return np.floor(value + 0.5).astype(np.int16)

    # ------------------------------------------------------------------------
    # loadModel
    # ------------------------------------------------------------------------
    @staticmethod
    def _loadModel(modelFile: Path):
        """
        A joblib model file, or a CompactForest directory (the same trees in compact form, identical predictions;
        Reports/Smaller_RF_Models_Compact_And_Subsample_Plan.md).  A compact model is memory-mapped by default: map
        tasks on one node share its pages through the file cache, and its pages are not private memory
        (Reports/Map_Memory_Proof_Results.md).  VCF_COMPACT_MMAP=0 reads it into memory instead.
        """
        if CompactForest.isCompact(modelFile):
            return CompactForest.load(modelFile, mmap=os.environ.get('VCF_COMPACT_MMAP', '1') != '0')

        with open(modelFile, 'rb') as f:
            return joblib.load(f)

    # ------------------------------------------------------------------------
    # members
    # ------------------------------------------------------------------------
    @staticmethod
    def _members(model) -> list:
        """
        The ensemble members as (estimator, feature indices or None), or None if the model has none.
        Random Forest: estimators_.  BaggingRegressor: estimators_ with estimators_features_, because each
        member sees only its own subset of the predictors.  XGBoost has no members.
        """
        ests = getattr(model, 'estimators_', None)

        if not isinstance(ests, list) or not ests:
            return None

        feats = getattr(model, 'estimators_features_', None)

        return list(zip(ests, feats if feats is not None else [None] * len(ests)))

    # ------------------------------------------------------------------------
    # combine
    # ------------------------------------------------------------------------
    @staticmethod
    def _combine(P: np.ndarray, agg: str) -> np.ndarray:
        """Combine the members' predictions P (members x pixels) per pixel."""
        if agg == 'mean':
            return P.mean(axis=0)

        if agg == 'median':
            return np.median(P, axis=0)

        if agg == 'mid20':

            k = VcfPredict.MID20_TRIM

            if P.shape[0] <= 2 * k:
                raise ValueError('mid20 needs more than ' + str(2 * k) + ' members; the model has ' +
                                 str(P.shape[0]))

            return np.sort(P, axis=0)[k:P.shape[0] - k].mean(axis=0)

        raise ValueError('Unknown aggregation ' + str(agg))

    # ------------------------------------------------------------------------
    # memberPredict
    # ------------------------------------------------------------------------
    @staticmethod
    def _memberPredict(model, X: np.ndarray, agg: str, nJobs: int = 1) -> tuple:
        """
        Predict every member on X in blocks of rows, each member clipped to 0-100 as the C code does for its
        30 models (MOD44B3.c).  Returns (value, sd): value combines the members by agg, and sd is their
        population SD (ddof = 0), without the C code's +0.5 offset (Reports/SD_Layer_Plan.md).  The SD is the
        spread of the members whichever way the value is combined.
        """
        members = VcfPredict._members(model)
        value = np.empty(len(X), dtype=np.float64)
        sd = np.empty(len(X), dtype=np.float64)
        n = VcfPredict.PREDICT_CHUNK_ROWS

        def predictMember(member, xc):
            est, feats = member
            return np.clip(est.predict(xc if feats is None else xc[:, feats]), 0, 100)

        with ThreadPoolExecutor(nJobs) as ex:

            for i in range(0, len(X), n):

                xc = X[i:i + n]
                P = np.vstack(list(ex.map(lambda m: predictMember(m, xc), members)))
                value[i:i + n] = VcfPredict._combine(P, agg)
                sd[i:i + n] = P.std(axis=0)
                del P

        return value, sd

    # ------------------------------------------------------------------------
    # runTileForYear
    # ------------------------------------------------------------------------
    def runTileForYear(self, tid: str, year: int) -> list[np.ndarray]:

        self._logger.info('Running ' + tid + ' for ' + str(year))
        
        pctTc, sdTc = self._runPrediction(tid,
                                          year,
                                          self._treeCoverRf,
                                          self._treeCoverTopN,
                                          self._treeCoverAgg)

        pctNonvegRaw, sdNonveg = self._runPrediction(tid,
                                                     year,
                                                     self._nonvegRf,
                                                     self._nonvegTopN,
                                                     self._nonvegAgg)

        # ---
        # Tree cover, non-vegetated and non-tree vegetation must sum to exactly 100 on an
        # ordinary pixel.  If tree cover and non-vegetated overshoot 100 together, reduce
        # non-vegetated - never tree cover - down to what is left after tree cover, so
        # non-tree vegetation lands on exactly 0 by arithmetic rather than a separate floor.
        # Water (mask == MASK_WATER) and out-of-projection (mask == MASK_OUT_OF_PROJ) pixels
        # are excluded from this reduction: they already carry 200/200 or 250/250 from
        # _runPrediction's _maskPrediction call, and applying the reduction there would
        # corrupt them (e.g. water: 100 - 200 = -100).
        # ---
        mask: np.ndarray = self._getMask(tid, year)
        okMask = (mask != VcfPredict.MASK_WATER) & (mask != VcfPredict.MASK_OUT_OF_PROJ)

        # ---
        # Blend A (BlendA.blend) on rows v06-v11 when a partner is given.  The
        # partner reads its own (V4r, unsubstituted) metrics.  The tree SD
        # layer stays the clear model's.  The sum cap below then uses the
        # blended tree, as blend/blend_year.py --rawBare did.
        # ---
        chain = dict(tile=tid, year=year, blend=False, bp10=None)

        if self._partnerRf is not None and BlendA.inScope(VcfPredict._vTile(tid)):

            self._logger.info('Blend A: predicting the partner for ' + tid)

            partner, _ = self._runPrediction(tid,
                                             year,
                                             self._partnerRf,
                                             self._partnerTopN,
                                             self._partnerAgg,
                                             metricsDir=self._partnerMetricsDir,
                                             lowSun='off')

            pctTc, _, stats = BlendA.blend(pctTc, partner, BlendA.loadCounts(self._countsDir, tid))
            chain.update(blend=True, **stats)
            del partner

        pctNonveg: np.ndarray = np.where(okMask,
                                         np.minimum(pctNonvegRaw, 100 - pctTc),
                                         pctNonvegRaw)

        # ---
        # High-latitude rule: on v00 tiles (north of about 80 N) every land pixel is tree
        # cover 0 and non-vegetated 100.  There are no trees there and very likely no
        # vegetation, and two of the eight composite periods have no input at all, which
        # leaves the models with spurious tree cover.  This is the C code's Greenland rule
        # (MOD44B3.c), which covers all land on these tiles.  The SD layers keep the
        # members' spread, as in the C code.
        # ---
        if VcfPredict._vTile(tid) in VcfPredict.HIGH_LATITUDE_BARE_VTILES:

            self._logger.info('High-latitude rule: ' + tid + ' land set to tree 0, non-vegetated 100.')
            pctTc = np.where(okMask, 0, pctTc).astype(pctTc.dtype)
            pctNonveg = np.where(okMask, 100, pctNonveg).astype(pctNonveg.dtype)

        pctNontreeVeg: np.ndarray = 100 - pctTc - pctNonveg
        pctNontreeVeg = self._maskPrediction(tid, year, pctNontreeVeg)

        # BP10 (BlendA.bp10) on the final layers, from the uncapped non-vegetated value.
        if self._bp10:

            pctTc, pctNonveg, pctNontreeVeg, chain['bp10'] = \
                BlendA.bp10(pctTc, pctNonveg, pctNontreeVeg, pctNonvegRaw)

        if chain['blend'] or self._bp10:
            (Path(self._outDir) / (tid + '-' + str(year) + '-chain.json')).write_text(json.dumps(chain))

        self._write(tid, year, 'Percent_Tree_Cover', pctTc)
        self._write(tid, year, 'Percent_NonVegetated', pctNonveg)
        self._write(tid, year, 'Percent_NonTree_Vegetation', pctNontreeVeg)

        # The SD layers describe the members' disagreement before the sum cap above, as in the C code.
        if sdTc is not None:
            self._write(tid, year, 'Percent_Tree_Cover_SD', sdTc, VcfPredict.SD_DATA_TYPE)

        if sdNonveg is not None:
            self._write(tid, year, 'Percent_NonVegetated_SD', sdNonveg, VcfPredict.SD_DATA_TYPE)

        # ---
        # Quality and Cloud, as in the historical MOD44B: one bit per composite
        # period (period 1 in bit 0).  Quality: the period fell back to the mean
        # of all its days; Cloud: it fell back and no day was clear.  0 on water
        # and out-of-projection pixels.  No no-data value: every byte is a valid
        # set of flags.
        # ---
        quality, cloud = Metrics(tid,
                                 year,
                                 self._productType,
                                 self._metricsDir,
                                 self._logger).getPeriodFlags()

        self._write(tid, year, 'Quality', np.where(okMask, quality, 0), noData=False)
        self._write(tid, year, 'Cloud', np.where(okMask, cloud, 0), noData=False)

        return pctTc, pctNonveg, pctNontreeVeg
        
    # ------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------
    def run(self, tids: list[str], years: list[int]) -> \
        list[list[np.ndarray]]:
        
        results = [self.runTileForYear(t, y) for t in tids for y in years]
        return results

    # ------------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------------
    def _write(self, 
               tid: str, 
               year: int, 
               outPrefix: str, 
               raster: np.ndarray,
               dataType: int = PCT_DATA_TYPE,
               noData: bool = True) -> Path:
        
        fName = outPrefix + '-' + tid + '-' + str(year)
        outName = Path(self._outDir) / (fName + '.tif')
        
        # Refuse values the storage type cannot hold rather than letting them wrap.
        npType = gdal_array.GDALTypeCodeToNumericTypeCode(dataType)
        info = np.iinfo(npType)

        if raster.min() < info.min or raster.max() > info.max:

            raise ValueError(fName + ': values ' + str(raster.min()) + '-' + str(raster.max()) +
                             ' do not fit ' + gdal.GetDataTypeName(dataType))

        ds = gdal.GetDriverByName('GTiff').Create(str(outName),
                                                  raster.shape[1],
                                                  raster.shape[0],
                                                  1,
                                                  dataType,
                                                  options=['COMPRESS=LZW'])
        srs = osr.SpatialReference()
        srs.ImportFromProj4(VcfPredict.MODIS_SINU)
        ds.SetProjection(srs.ExportToWkt())
        ds.SetGeoTransform(VcfPredict._geoTransform(tid, raster.shape))
        band = ds.GetRasterBand(1)

        if dataType == VcfPredict.PCT_DATA_TYPE and noData:
            band.SetNoDataValue(VcfPredict.PCT_NO_DATA)

        band.WriteArray(raster.astype(npType))
        band = None
        ds = None

        return outName
        