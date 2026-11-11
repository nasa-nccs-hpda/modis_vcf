"""model/Mod44bH5.py: the Collection 7 product file written from layers and read back."""
from datetime import datetime, timezone
import re

import h5py
import numpy as np
import pytest

from modis_vcf.model import Mod44bH5 as H

N = H.SIZE


@pytest.fixture(scope='module')
def written(tmp_path_factory):
    d = tmp_path_factory.mktemp('h5')
    tree = np.zeros((N, N), np.uint8); tree[:10] = 37; tree[10:20] = 200; tree[20:30] = 250
    layers = {'Percent_Tree_Cover': tree, 'Percent_NonVegetated': np.full((N, N), 50, np.uint8),
              'Percent_NonTree_Vegetation': np.full((N, N), 13, np.uint8), 'Percent_Tree_Cover_SD': np.full((N, N), 3632, np.uint16),
              'Quality': np.full((N, N), 5, np.uint8), 'Cloud': np.zeros((N, N), np.uint8)}      # no non-vegetated SD
    prod = datetime(2026, 11, 11, 8, 0, 0, tzinfo=timezone.utc)
    path = H.write(d, 'h12v02', 2019, layers, inputs=['MOD44CQ.A2019065.h12v02.007.x.h5'],
                   overrides={'PGEVersion': '9.9.9'}, production=prod)
    return d, path, layers


def test_name_and_find(written):
    d, path, _ = written
    assert re.fullmatch(r'MOD44B\.A2019065\.h12v02\.%s\.2026315080000\.h5' % H.COLLECTION, path.name)
    assert H.find(d, 'h12v02', 2019) == path and not path.with_name(path.name + '.part').exists()


def test_layers_round_trip(written):
    d, path, layers = written
    for f, a in layers.items():
        assert np.array_equal(H.readLayer(path, f), a)
    assert not H.readLayer(path, 'Percent_NonVegetated_SD').any()            # a missing SD layer is its fill, 0
    assert np.array_equal(H.readMapLayer(d, 'h12v02', 2019, 'Quality'), layers['Quality'])


def test_attributes(written):
    _, path, _ = written
    with h5py.File(path, 'r') as f:
        a = dict(f.attrs)
        d = f[H.FIELDS_PATH + '/Percent_Tree_Cover']
        assert d.dtype == np.uint8 and int(d.attrs['_FillValue']) == H.PCT_FILL
    s = lambda v: v.decode() if isinstance(v, bytes) else str(v)
    assert s(a['HORIZONTALTILENUMBER']) == '12' and s(a['VERTICALTILENUMBER']) == '2' and s(a['PGEVersion']) == '9.9.9'
    assert s(a['InputDataProducts']) == 'MOD44CQ.A2019065.h12v02.007.x.h5'
    assert float(a['geospatial_lat_max']) > float(a['geospatial_lat_min'])


def test_refuses_values_out_of_range(tmp_path):
    with pytest.raises(ValueError):
        H.write(tmp_path, 'h12v02', 2019, {'Percent_Tree_Cover': np.full((N, N), 300, np.int16)})
