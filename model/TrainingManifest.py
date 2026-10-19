from concurrent.futures import ProcessPoolExecutor
import datetime
import hashlib
import json
import logging
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from modis_vcf.model.Metrics import Metrics


# ----------------------------------------------------------------------------
# TrainingManifest
#
# The training sample of a model version as a manifest of keys (v1.4.1 P4):
# one row per training pixel with its part (Train / Test), its row id in the
# split, its tile-year, raster row (x) and column (y), and its label.  The
# predictor columns are not stored: build() computes them from a metrics
# root, as the original splits were made, so a split can be rebuilt from the
# manifest whenever the metrics can be rebuilt.
#
#   extract(split dir) -> manifest   the sample of an existing split
#   sample(label files) -> manifest  the sample drawn from the label rasters
#                                    (fresh-sample mode)
#   build(manifest)    -> split dir  xTrain / xTest / yTrain / yTest /
#                                    keysTrain / keysTest, in manifest order
#
# Keys: x = raster row, y = column (arr[x, y]).  The h16v02 ice rows of the
# bare sample are rows like any other; each version re-samples them, so each
# version has its own manifest.  Manifests are written next to each version's
# models; the current product's is also kept in the package (data/).
# ----------------------------------------------------------------------------
PARTS = ('Train', 'Test')
COLUMNS = ['part', 'row', 'tid-year', 'x', 'y', 'label']
FORMAT = 'TrainingManifest 1'


# ----------------------------------------------------------------------------
# _sha256
# ----------------------------------------------------------------------------
def _sha256(f: Path) -> str:

    h = hashlib.sha256()

    with open(f, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)

    return h.hexdigest()


# ----------------------------------------------------------------------------
# extract
# ----------------------------------------------------------------------------
def extract(splitDir: Path, outFile: Path, version: str, notes: str = '') -> dict:
    """Write <outFile> (parquet) and <outFile>.json from a split directory."""

    splitDir, outFile = Path(splitDir), Path(outFile)

    if outFile.exists():
        raise RuntimeError(str(outFile) + ' exists')

    frames, meta = [], dict(format=FORMAT, version=version, source=str(splitDir), notes=notes,
                            created=datetime.datetime.now().isoformat(timespec='seconds'),
                            source_sha256={}, parts={})

    for part in PARTS:

        keys = pd.read_parquet(splitDir / ('keys' + part + '.parq'))
        y = pd.read_parquet(splitDir / ('y' + part + '.parq'))

        if len(keys) != len(y) or not keys.index.equals(y.index) or y.shape[1] != 1:
            raise RuntimeError(part + ': keys and labels do not line up in ' + str(splitDir))

        label = y.columns[0]
        meta['trainingType'] = label
        indexKind = 'range' if isinstance(keys.index, pd.RangeIndex) and keys.index.start == 0 \
            and keys.index.step == 1 else 'stored'

        df = pd.DataFrame({'part': part, 'row': keys.index.to_numpy(np.int64),
                           'tid-year': keys['tid-year'].to_numpy(), 'x': keys.x.to_numpy(),
                           'y': keys.y.to_numpy(), 'label': y[label].to_numpy()})
        frames.append(df)

        tiles = df['tid-year'].str[:6].value_counts().sort_index()
        meta['parts'][part] = dict(rows=len(df), index=indexKind, label_dtype=str(y[label].dtype),
                                   rows_per_tile={k: int(v) for k, v in tiles.items()})

        for k in ('keys', 'y'):
            f = splitDir / (k + part + '.parq')
            meta['source_sha256'][f.name] = _sha256(f)

    m = pd.concat(frames, ignore_index=True)[COLUMNS]
    m.to_parquet(outFile, index=False)
    meta['sha256'] = _sha256(outFile)
    Path(str(outFile) + '.json').write_text(json.dumps(meta, indent=1))
    return meta


# ----------------------------------------------------------------------------
# sample
#
# Fresh-sample mode: the sample drawn from the label rasters by the rule the
# product's splits were made with.  Every pixel of a label file (uint8,
# 4800 x 4800, row-major) whose value is not noData is a row: key (tid-year,
# x = row, y = column), label as int16.  Rows are sorted by key (stable) and
# a row is held out (Test) when the pandas hash of its key is 0 mod 5.
# Extra rows (a parquet with tid-year, x, y, label: the h16v02 ice points,
# label 100) are a label input too; they are split by the same hash and
# appended after the label-file rows of each part, in their file order.
# Row ids: 'range' numbers each part from 0; 'position' keeps each label-file
# row's position in the sorted set (extra rows continue after the last one).
# ----------------------------------------------------------------------------
KEY_DTYPES = {'tid-year': object, 'x': np.int64, 'y': np.int64}


def _testRows(keys: pd.DataFrame) -> np.ndarray:
    return (pd.util.hash_pandas_object(keys[['tid-year', 'x', 'y']], index=False) % 5 == 0).to_numpy()


def sample(labelFiles: dict, trainingType: str, year: int, outFile: Path, version: str, extra: Path = None,
           index: str = 'range', noData: int = 255, notes: str = '') -> dict:
    """Write <outFile> (parquet) and <outFile>.json from label files {tile: path}."""

    outFile = Path(outFile)

    if outFile.exists():
        raise RuntimeError(str(outFile) + ' exists')

    if index not in ('range', 'position'):
        raise ValueError('index must be range or position')

    frames, sources = [], {}

    for tile in sorted(labelFiles):

        f = Path(labelFiles[tile])
        a = np.fromfile(f, np.uint8)

        if a.size != 4800 * 4800:
            raise RuntimeError(str(f) + ': ' + str(a.size) + ' values, not 4800 x 4800')

        a = a.reshape(4800, 4800)
        x, y = np.nonzero(a != noData)
        frames.append(pd.DataFrame({'tid-year': tile + '-' + str(year), 'x': x, 'y': y,
                                    'label': a[x, y].astype(np.int16)}))
        sources[tile] = dict(file=str(f), sha256=_sha256(f))

    base = pd.concat(frames, ignore_index=True).astype(KEY_DTYPES)
    base = base.sort_values(['tid-year', 'x', 'y'], kind='mergesort', ignore_index=True)
    base['test'] = _testRows(base)
    base['row'] = np.arange(len(base), dtype=np.int64)

    if extra is not None:

        e = pd.read_parquet(extra)[['tid-year', 'x', 'y', 'label']].astype(KEY_DTYPES)
        e['label'] = e['label'].astype(np.int16)
        e['test'] = _testRows(e)
        e['row'] = len(base) + np.arange(len(e), dtype=np.int64)
        sources['extra'] = dict(file=str(extra), sha256=_sha256(Path(extra)), rows=len(e))
        both = pd.concat([base, e], ignore_index=True)

        if both.duplicated(['tid-year', 'x', 'y']).any():
            raise RuntimeError('extra rows repeat label-file keys')

    else:
        e = base.iloc[:0]

    frames, meta = [], dict(format=FORMAT, version=version, source='labels', notes=notes, trainingType=trainingType,
                            created=datetime.datetime.now().isoformat(timespec='seconds'), year=year,
                            rule=dict(noData=noData, order='key, stable', test='pandas hash of key % 5 == 0',
                                      extra='appended per part', index=index),
                            labels=sources, parts={})

    for part, te in (('Train', False), ('Test', True)):

        p = pd.concat([base[base.test == te], e[e.test == te]], ignore_index=True)
        row = np.arange(len(p), dtype=np.int64) if index == 'range' else p.row.to_numpy(np.int64)
        df = pd.DataFrame({'part': part, 'row': row, 'tid-year': p['tid-year'].to_numpy(), 'x': p.x.to_numpy(),
                           'y': p.y.to_numpy(), 'label': p.label.to_numpy()})
        frames.append(df)
        tiles = df['tid-year'].str[:6].value_counts().sort_index()
        meta['parts'][part] = dict(rows=len(df), index='range' if index == 'range' else 'stored', label_dtype='int16',
                                   rows_per_tile={k: int(v) for k, v in tiles.items()})

    m = pd.concat(frames, ignore_index=True)[COLUMNS]
    m.to_parquet(outFile, index=False)
    meta['sha256'] = _sha256(outFile)
    Path(str(outFile) + '.json').write_text(json.dumps(meta, indent=1))
    return meta


# ----------------------------------------------------------------------------
# read
# ----------------------------------------------------------------------------
def read(manifestFile: Path) -> tuple:

    meta = json.loads(Path(str(manifestFile) + '.json').read_text())

    if meta.get('format') != FORMAT:
        raise RuntimeError(str(manifestFile) + ': not a ' + FORMAT)

    return pd.read_parquet(manifestFile), meta


# ----------------------------------------------------------------------------
# _tileFeatures
#
# The predictors of one tile's rows, Train and Test in one pass (one Metrics,
# so each metric file is read once).  Runs in a worker process when build()
# has workers > 1; the low-sun mode reaches it through the environment.
# ----------------------------------------------------------------------------
def _tileFeatures(productType, tid: str, year: int, root: Path, predictors: list, rowsByPart: dict,
                  lowSun: str = None, logLevel: int = logging.WARNING) -> dict:

    logger = logging.getLogger('trainingManifest.' + tid)
    logger.setLevel(logLevel)
    mt = Metrics(tid, year, productType, Path(root), logger, lowSun=lowSun)
    out = mt.getRowsFromRf(predictors, rowsByPart)
    return {part: {n: out[part][n] for n in predictors} for part in rowsByPart}


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------
def build(manifestFile: Path, predictors: list, rootFor, year: int, productType, outDir: Path,
          logger: logging.Logger = None, tiles: list = None, keysOnly: bool = False,
          lowSun: str = None, workers: int = 1) -> dict:
    """Write a split from a manifest.  rootFor(tile) -> metrics root.  tiles limits the rows to those
    tiles (in manifest order); keysOnly writes keys and labels only (no metrics read).  workers > 1
    computes the tiles in that many processes; the split is the same for any number (S5)."""

    logger = logger or logging.getLogger()
    m, meta = read(manifestFile)
    outDir = Path(outDir)

    if outDir.exists():
        raise RuntimeError(str(outDir) + ' exists')

    outDir.mkdir(parents=True)
    label = meta['trainingType']
    report, parts = {}, {}

    for part in PARTS:

        p = m[m.part == part]
        tid = p['tid-year'].str[:6].to_numpy()

        if tiles is not None:
            p = p[np.isin(tid, tiles)]
            tid = p['tid-year'].str[:6].to_numpy()

        index = pd.Index(p.row.to_numpy(np.int64))

        if meta['parts'][part]['index'] == 'range' and tiles is None:
            index = pd.RangeIndex(len(p))

        keys = pd.DataFrame({'tid-year': p['tid-year'].to_numpy(), 'x': p.x.to_numpy(), 'y': p.y.to_numpy()},
                            index=index)
        y = pd.DataFrame({label: p.label.to_numpy().astype(meta['parts'][part]['label_dtype'])}, index=index)
        keys.to_parquet(outDir / ('keys' + part + '.parq'))
        y.to_parquet(outDir / ('y' + part + '.parq'))
        report[part] = dict(rows=len(p))

        if not keysOnly:
            parts[part] = (p, tid, index)

    if not keysOnly:

        order = list(pd.unique(np.concatenate([parts[q][1] for q in parts])))
        jobs = {}

        for t in order:
            rows = {}
            for q, (p, tid, _) in parts.items():
                sel = np.flatnonzero(tid == t)
                if len(sel):
                    rows[q] = (p.x.to_numpy()[sel], p.y.to_numpy()[sel])
            jobs[t] = rows

        cols = {q: {} for q in parts}

        def store(t, res):
            for q, feats in res.items():
                sel = np.flatnonzero(parts[q][1] == t)
                for n, v in feats.items():
                    if n not in cols[q]:
                        cols[q][n] = np.empty(len(parts[q][0]), dtype=v.dtype)
                    cols[q][n][sel] = v

        if workers <= 1:

            for t in order:
                logger.info('Training rows: ' + t + ' ' + str({q: len(r[0]) for q, r in jobs[t].items()}))
                store(t, _tileFeatures(productType, t, year, rootFor(t), predictors, jobs[t], lowSun,
                                       logger.getEffectiveLevel()))

        else:

            logger.info('Training rows: ' + str(len(order)) + ' tiles in ' + str(workers) + ' processes')

            ex = ProcessPoolExecutor(min(workers, len(order)), mp_context=multiprocessing.get_context('spawn'))

            try:

                futures = {t: ex.submit(_tileFeatures, productType, t, year, rootFor(t), predictors, jobs[t], lowSun)
                           for t in order}

                for t in order:
                    store(t, futures[t].result())
                    logger.info('Training rows: ' + t + ' done')

            except BaseException:

                # A stop or a failure: drop the tiles not started and end the workers now, so that no worker
                # outlives the split (shutdown(wait=True) would wait for the running tiles).
                for proc in list((getattr(ex, '_processes', None) or {}).values()):
                    proc.terminate()

                ex.shutdown(wait=False, cancel_futures=True)
                raise

            ex.shutdown()

        for q, (p, _, index) in parts.items():
            pd.DataFrame({n: cols[q][n] for n in predictors}, index=index).to_parquet(outDir / ('x' + q + '.parq'))

    (outDir / 'manifest.json').write_text(json.dumps(dict(manifest=str(manifestFile), manifest_sha256=_sha256(manifestFile),
                                                          year=year, tiles=tiles, keysOnly=keysOnly, workers=workers,
                                                          report=report),
                                                     indent=1))
    return report
