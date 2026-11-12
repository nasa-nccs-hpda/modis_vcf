"""Shared set-up for the unit tests (run `pytest` from the repository root, inside the modis_vcf environment).

- The package is imported as `modis_vcf`, so the repository folder must be named modis_vcf; its parent folder is put on sys.path.
- Tests marked `integration` read the NCCS ADAPT input data (MOD44C, MOD44W) and take minutes; they are skipped unless
  VCF_INTEGRATION=1.  Every other test uses small synthetic data written to temporary folders and runs anywhere.
- The input-folder variables are cleared for each test, so a test never reads a real input by accident.
"""
import os
import sys
from pathlib import Path

import pytest

from synthetic import smallTile, smallTileModule  # noqa: F401  (fixtures shared by the tests)

ROOT = Path(__file__).resolve().parents[1]

if ROOT.name != 'modis_vcf':
    raise RuntimeError('The repository folder must be named modis_vcf (it is imported as a package); it is ' + ROOT.name)

if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

INPUT_VARIABLES = ('VCF_MOD44_DIR', 'VCF_MOD44W_DIR', 'VCF_LOWSUN_T', 'VCF_LOWSUN_LE0', 'VCF_LOWSUN_CHECK', 'VCF_WRITE_DAILIES',
                   'VCF_COMPACT_MMAP')


def pytest_collection_modifyitems(config, items):

    if os.environ.get('VCF_INTEGRATION') == '1':
        return

    skip = pytest.mark.skip(reason='integration test (ADAPT data); set VCF_INTEGRATION=1 to run')

    for item in items:
        if 'integration' in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _clearCaches():
    """CompositeDayFile caches the QA masks of an input day per product, tile, year and day (one input folder per process in
    production); tests use many folders for the same tile-year, so the cache is cleared for each test."""

    from modis_vcf.model.CompositeDayFile import CompositeDayFile
    CompositeDayFile._maskCache.clear()
    yield
    CompositeDayFile._maskCache.clear()


@pytest.fixture(autouse=True)
def _cleanEnvironment(monkeypatch, request):
    """Clear the input and low-sun variables for every synthetic test (integration tests keep the caller's environment)."""

    if 'integration' not in request.keywords:
        for v in INPUT_VARIABLES:
            monkeypatch.delenv(v, raising=False)
