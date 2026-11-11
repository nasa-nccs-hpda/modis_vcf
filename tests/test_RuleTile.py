"""model/RuleTile.py: which tiles are mapped by rule."""
from modis_vcf.model import RuleTile


def test_rule_rows():
    assert RuleTile.isRuleTile('h16v15') and RuleTile.isRuleTile('h20v17')
    assert not RuleTile.isRuleTile('h16v14') and not RuleTile.isRuleTile('h12v02') and not RuleTile.isRuleTile('h17v00')
