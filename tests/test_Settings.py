"""model/Settings.py: input folders from the environment, no defaults; collections."""
import importlib
from pathlib import Path

from modis_vcf.model import Settings


def _reload(monkeypatch, **env):
    for k in ('VCF_MOD44_DIR', 'VCF_MOD44W_DIR', 'VCF_MOD44_COLLECTION', 'VCF_OUTPUT_COLLECTION'):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(Settings)


def test_no_default_folders(monkeypatch):
    s = _reload(monkeypatch)
    assert s.MOD44_DIR is None and s.MOD44W_DIR is None
    assert s.MOD44_COLLECTION == '7' and s.OUTPUT_COLLECTION == '071'
    _reload(monkeypatch)


def test_from_environment(monkeypatch, tmp_path):
    s = _reload(monkeypatch, VCF_MOD44_DIR=str(tmp_path), VCF_MOD44W_DIR=str(tmp_path / 'w'), VCF_MOD44_COLLECTION='61')
    assert s.MOD44_DIR == Path(tmp_path) and s.MOD44W_DIR == tmp_path / 'w' and s.MOD44_COLLECTION == '61'
    _reload(monkeypatch)


def test_missing_folder_named_in_error():
    import pytest
    from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
    from modis_vcf.model.ProductTypeMod44W import ProductTypeMod44W
    with pytest.raises(ValueError, match='VCF_MOD44_DIR'):
        ProductTypeMod44(None)
    with pytest.raises(ValueError, match='VCF_MOD44W_DIR'):
        ProductTypeMod44W(None)
