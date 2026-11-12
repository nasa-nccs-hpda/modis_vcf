"""model/YearRun.py: the year's task graph, task commands and spec checks, on fake manifests and MOD44W files (no task is run)."""
import json
import subprocess

import pytest

from modis_vcf.model import Settings
from modis_vcf.model.YearRun import YearRun

TILES = ['h12v02', 'h21v07', 'h16v16']               # a training tile, a blend tile, a rule tile


def _manifest(d, name, tiles):
    m = d / (name + '.parq')
    (d / (name + '.parq.json')).write_text(json.dumps({'parts': {'Train': {'rows_per_tile': {t: 10 for t in tiles}}}}))
    return str(m)


def _model(d, name):
    cf = d / (name + '.cf')
    cf.mkdir()
    (cf / 'meta.json').write_text('{}')
    (cf / 'feature.npy').write_bytes(b'x')
    (d / (name + 'TopN.bin')).write_bytes(b'top')
    subprocess.run('sha256sum %s.cf/feature.npy %s.cf/meta.json %sTopN.bin > %s.sha256' % (name, name, name, name),
                   shell=True, cwd=d, check=True)
    return {'model': str(cf), 'topN': str(d / (name + 'TopN.bin')), 'sha256': str(d / (name + '.sha256'))}


@pytest.fixture
def spec(tmp_path, monkeypatch):
    w = tmp_path / 'mod44w' / '2019' / '001'
    w.mkdir(parents=True)
    for t in TILES[:2]:                                  # h16v16 has no MOD44W: skipped, not failed
        (w / ('MOD44W.A2019001.%s.061.2020001.hdf' % t)).write_bytes(b'')
    for t in ('h17v15',):
        (w / ('MOD44W.A2019001.%s.061.2020001.hdf' % t)).write_bytes(b'')
    monkeypatch.setattr(Settings, 'MOD44W_DIR', tmp_path / 'mod44w')
    fit = {'nEstimators': 60, 'maxFeatures': 'sqrt', 'nJobs': 12, 'randomState': 0, 'minSamplesLeaf': 1}
    s = {'year': 2019, 'runRoot': str(tmp_path / 'run'), 'lowSun': 'le0', 'mapFormat': 'h5',
         'models': {'tree': {'manifest': _manifest(tmp_path, 'tree', ['h12v02']), 'predictors': 'p_tree.txt',
                             'trainingType': 'pcttree', 'fit': fit},
                    'bare': {'trainingType': 'pctbare', 'fixed': _model(tmp_path, 'pooled')}},
         'cloudFallbackTree': {'tiles': ['h21v07'], 'manifest': _manifest(tmp_path, 'tree', ['h12v02']),
                               'predictors': 'p_tree.txt', 'trainingType': 'pcttree', 'fit': fit},
         'mapTiles': ['h12v02', 'h21v07', 'h16v16', 'h17v15'], 'workers': {'metrics': 8, 'split': 6, 'mapJobs': 8, 'cloudFallback': 8}}
    return tmp_path, s


def _yr(d, s, name='spec.json'):
    (d / name).write_text(json.dumps(s))
    return YearRun(d / name)


def test_task_graph(spec):
    d, s = spec
    y = _yr(d, s)
    assert 'map:h16v16' not in y.tasks                                    # no MOD44W
    assert set(y.tasks) == {'metrics:h12v02', 'metrics:h21v07', 'split:tree', 'fit:tree', 'split:cloudFallbackTree',
                            'fit:cloudFallbackTree', 'map:h12v02', 'map:h21v07', 'map:h17v15'}
    assert y.deps['map:h17v15'] == []                                     # rule tile: no metrics, no models
    assert y.deps['map:h21v07'] == ['metrics:h21v07', 'fit:tree', 'fit:cloudFallbackTree']
    assert y.deps['map:h12v02'] == ['metrics:h12v02', 'fit:tree']         # fixed bare model: no split or fit
    assert y.cloudFallbackCompositeTiles == {'h21v07', 'h12v02'}          # blend tiles + the fitted model's training tiles


def test_commands(spec):
    d, s = spec
    y = _yr(d, s)
    (cmd,) = y._commands('map:h21v07')[:1]
    assert '--cloudFallbackTree' in cmd and '--lowSun' in cmd and cmd[cmd.index('--nv') + 1] == s['models']['bare']['fixed']['model']
    split = y._commands('split:cloudFallbackTree')[0]
    assert split[split.index('--lowSun') + 1] == 'off' and split[split.index('--root') + 1].endswith('cloudfallback')
    assert y._commands('map:h17v15')[0][-4:] == ['-y', '2019', '--format', 'h5']


def test_fixed_model_checksum(spec):
    d, s = spec
    y = _yr(d, s)
    y.checkFixedModels()
    (d / 'pooled.cf' / 'feature.npy').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='does not match'):
        y.checkFixedModels()
    (d / 'pooled.cf' / 'feature.npy').write_bytes(b'x')
    (d / 'pooled.cf' / 'extra.npy').write_bytes(b'')
    with pytest.raises(RuntimeError, match='does not list'):
        y.checkFixedModels()


def test_split_only(spec):
    d, s = spec
    s['models']['tree']['splitOnly'] = True
    s.pop('cloudFallbackTree')
    s['mapTiles'] = []
    y = _yr(d, s)
    assert sorted(y.tasks) == ['metrics:h12v02', 'split:tree']
    s['mapTiles'] = ['h12v02']
    with pytest.raises(ValueError, match='splitOnly'):
        _yr(d, s, 'spec2.json')


def test_old_keys_refused(spec):
    d, s = spec
    s['partner'] = s.pop('cloudFallbackTree')
    with pytest.raises(ValueError, match='cloudFallbackTree'):
        _yr(d, s)
