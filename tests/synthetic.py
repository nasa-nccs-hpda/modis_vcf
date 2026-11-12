"""Synthetic inputs for the unit tests: small fake Collection 7 MOD44CH / MOD44CQ files and helpers.

The package works on 4800 x 4800 tiles; the tests shrink the tile with the `smallTile` fixture (ProductType.ROWS / COLS and
CloudFallback.ROWS / COLS patched to SIZE) and write fake inputs of that size, so a whole processing year runs in seconds.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest

SIZE = 6
QC_GOOD = 32768                       # MOD44C state bit 15 (PGE72 overall QC); bits 0-1 = 0 (clear)
CQ_BANDS = ('Band_1', 'Band_2', 'solar_zenith', 'state')
CH_BANDS = ('Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7', 'Band_31')
YEAR_ONE_DAYS = list(range(65, 365, 16))   # DOY 065 ... 353
YEAR_TWO_DAYS = list(range(1, 58, 16))     # DOY 001 ... 049


def inputDays(year: int) -> list:
    """The 23 (year, day) inputs of a processing year."""
    return [(year, d) for d in YEAR_ONE_DAYS] + [(year + 1, d) for d in YEAR_TWO_DAYS]


def defaultValue(year: int, day: int, band: str) -> np.ndarray:
    """A clear, QC-good day with sun at 30 degrees and band values that differ by band and day."""
    if band == 'state':
        return np.full((SIZE, SIZE), QC_GOOD, np.uint16)
    if band == 'solar_zenith':
        return np.full((SIZE, SIZE), 30, np.int16)
    k = (CQ_BANDS + CH_BANDS).index(band)
    return np.full((SIZE, SIZE), 1000 + 100 * k + day, np.int16)


def writeInputs(root: Path, tile: str, year: int, value=defaultValue, skip=()) -> Path:
    """Write MOD44CQ and MOD44CH .h5 files for every input day of the processing year under <root>/<year>/ (days in skip left
    out, as missing granules).  value(year, day, band) gives each (SIZE, SIZE) array.  Returns root."""
    root = Path(root)
    for y, d in inputDays(year):
        if (y, d) in skip:
            continue
        (root / str(y)).mkdir(parents=True, exist_ok=True)
        for prefix, bands in (('CQ', CQ_BANDS), ('CH', CH_BANDS)):
            f = root / str(y) / ('MOD44%s.A%d%03d.%s.007.2026001000000.h5' % (prefix, y, d, tile))
            with h5py.File(f, 'w') as h:
                g = h.create_group('Data_Fields')
                for b in bands:
                    g.create_dataset(b, data=value(y, d, b))
    return root


@pytest.fixture
def smallTile(monkeypatch):
    """Shrink the tile to SIZE x SIZE pixels for one test."""
    from modis_vcf.model import CloudFallback
    from modis_vcf.model.ProductType import ProductType
    monkeypatch.setattr(ProductType, 'ROWS', SIZE)
    monkeypatch.setattr(ProductType, 'COLS', SIZE)
    monkeypatch.setattr(CloudFallback, 'ROWS', SIZE)
    monkeypatch.setattr(CloudFallback, 'COLS', SIZE)
    return SIZE


@pytest.fixture(scope='module')
def smallTileModule():
    """smallTile for a whole test module (module-scoped fixtures that build inputs and metrics once)."""
    from modis_vcf.model import CloudFallback
    from modis_vcf.model.ProductType import ProductType
    mp = pytest.MonkeyPatch()
    for obj in (ProductType, CloudFallback):
        mp.setattr(obj, 'ROWS', SIZE)
        mp.setattr(obj, 'COLS', SIZE)
    yield SIZE
    mp.undo()


def metricsRoot(inputs: Path, root: Path, tile: str = 'h12v02', year: int = 2019, lowSun: str = 'le0'):
    """A Metrics object (low-sun mode lowSun) for synthetic inputs; root is created."""
    import logging
    from modis_vcf.model import LowSun
    from modis_vcf.model.Metrics import Metrics
    from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
    LowSun.configure(lowSun)
    Path(root).mkdir(parents=True, exist_ok=True)
    return Metrics(tile, year, ProductTypeMod44(Path(inputs)), Path(root), logging.getLogger('test'))
