import datetime
import hashlib
import json
import logging
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
# read
# ----------------------------------------------------------------------------
def read(manifestFile: Path) -> tuple:

    meta = json.loads(Path(str(manifestFile) + '.json').read_text())

    if meta.get('format') != FORMAT:
        raise RuntimeError(str(manifestFile) + ': not a ' + FORMAT)

    return pd.read_parquet(manifestFile), meta


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------
def build(manifestFile: Path, predictors: list, rootFor, year: int, productType, outDir: Path,
          logger: logging.Logger = None, tiles: list = None, keysOnly: bool = False,
          lowSun: str = None) -> dict:
    """Write a split from a manifest.  rootFor(tile) -> metrics root.  tiles limits the rows to those
    tiles (in manifest order); keysOnly writes keys and labels only (no metrics read)."""

    logger = logger or logging.getLogger()
    m, meta = read(manifestFile)
    outDir = Path(outDir)

    if outDir.exists():
        raise RuntimeError(str(outDir) + ' exists')

    outDir.mkdir(parents=True)
    label = meta['trainingType']
    report = {}

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

        if keysOnly:
            continue

        cols = {}
        xs, ys = p.x.to_numpy(), p.y.to_numpy()

        for t in pd.unique(tid):

            sel = np.flatnonzero(tid == t)
            logger.info('Training rows: ' + part + ' ' + t + ' ' + str(len(sel)))
            mt = Metrics(t, year, productType, Path(rootFor(t)), logger, lowSun=lowSun)

            for n in predictors:

                v = mt.getMetricFromRf(n)[xs[sel], ys[sel]]

                if n not in cols:
                    cols[n] = np.empty(len(p), dtype=v.dtype)

                cols[n][sel] = v

            del mt

        pd.DataFrame({n: cols[n] for n in predictors}, index=index).to_parquet(outDir / ('x' + part + '.parq'))

    (outDir / 'manifest.json').write_text(json.dumps(dict(manifest=str(manifestFile), manifest_sha256=_sha256(manifestFile),
                                                          year=year, tiles=tiles, keysOnly=keysOnly, report=report),
                                                     indent=1))
    return report
