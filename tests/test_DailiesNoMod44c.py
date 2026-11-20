"""v2.0.1: runs whose composites come from dailies never read MOD44C (ProductType noInput, NoMod44cInput)."""
import logging
from pathlib import Path

import numpy as np
import pytest

import synthetic as syn
from test_YearRun import spec, _yr                       # noqa: F401  (fixture)

from modis_vcf.model.Metrics import Metrics
from modis_vcf.model.ProductType import NoMod44cInput
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44


def _noInput():
    return ProductTypeMod44(None, noInput=True)


def test_folder_still_required_for_mod44c():
    with pytest.raises(ValueError, match='VCF_MOD44_DIR'):
        ProductTypeMod44(None)


def test_no_input_has_calendar_and_never_reads():
    pt, ref = _noInput(), ProductTypeMod44(Path('.'))
    assert pt.inputDir is None
    assert (pt.yearOneDays, pt.yearTwoDays, pt.dayStep) == (ref.yearOneDays, ref.yearTwoDays, ref.dayStep)
    assert not issubclass(NoMod44cInput, RuntimeError)   # the MOD44C path turns RuntimeError into an empty day
    with pytest.raises(NoMod44cInput, match='h12v02'):
        pt.findFile('h12v02', 2019, 65, ProductTypeMod44.BAND1)


def test_missing_composite_stops_instead_of_empty_day(smallTile, tmp_path):
    mt = Metrics('h12v02', 2019, _noInput(), tmp_path, logging.getLogger('test'))
    with pytest.raises(NoMod44cInput):
        mt.getBandCube('Band_1')


def test_cached_composites_read_without_input(smallTile, tmp_path):
    syn.writeInputs(tmp_path / 'in', 'h12v02', 2019)
    ref = syn.metricsRoot(tmp_path / 'in', tmp_path / 'metrics')
    a, xa = ref.getBandCube('Band_1')
    mt = Metrics('h12v02', 2019, _noInput(), tmp_path / 'metrics', logging.getLogger('test'))
    b, xb = mt.getBandCube('Band_1')
    assert xa == xb and np.array_equal(a, b)


def test_map_commands_pass_input(spec):                   # noqa: F811
    d, s = spec
    assert '--input' not in _yr(d, s)._commands('map:h12v02')[0]
    s['input'], s['dailiesDir'] = 'dailies', str(d)
    s.pop('cloudFallbackTree')
    c = _yr(d, s, 'spec2.json')._commands('map:h12v02')[0]
    assert c[c.index('--input') + 1] == 'dailies'
