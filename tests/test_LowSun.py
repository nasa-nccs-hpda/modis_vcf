"""model/LowSun.py: modes, the substitution mask and the environment it travels in."""
import os

import numpy as np
import pytest

from modis_vcf.model import LowSun


def test_default_and_modes():
    LowSun.configure(None)
    assert LowSun.settings() == (70, True) and LowSun.modeName() == 'le0'
    LowSun.configure('s70')
    assert LowSun.settings() == (70, False) and LowSun.modeName() == 's70'
    LowSun.configure('off')
    assert LowSun.settings()[0] == 0 and LowSun.modeName() == 'off' and not LowSun.appliesTo('h12v02')
    with pytest.raises(ValueError):
        LowSun.configure('bad')


def test_hand_set_values_replaced():
    os.environ[LowSun.ENV_T] = '0'
    os.environ[LowSun.ENV_LE0] = '0'
    LowSun.configure(None)                     # what every entry point does when --lowSun is not given
    assert LowSun.settings() == (70, True)


def test_mask():
    solz = np.array([-5, 0, 30, 70, 71])
    assert LowSun.mask(solz, 70, True).tolist() == [True, True, False, False, True]
    assert LowSun.mask(solz, 70, False).tolist() == [False, False, False, False, True]
    assert not LowSun.mask(solz, 0, False).any()
