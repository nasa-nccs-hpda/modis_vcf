"""The input sensors the dailies compositor (DailiesComposite) can read, one description each.

A description holds everything the compositor needs to know about a sensor's daily surface-reflectance inputs: the
product names, their collection, the HDF5 groups and dataset names, which bands enter the validity test, and the
sensor-specific band rules.  The compositing rules themselves (B2: good-day test, groups, fallback classes F1-F4,
polar-night fill) are the same for every sensor and live in DailiesComposite.

Terra and Aqua are described here.  Another sensor (e.g. VIIRS) would be a new description, plus an input reader if its
files are laid out differently from the MODIS L2G products; the compositing rules would not change.

Dailies folder layout (both sensors): <dailies root>/<PRODUCT>/<calendar year>/<PRODUCT>.A<yyyyddd>.<tile>.<collection>.*.h5
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Sensor:
    """One sensor's daily inputs.

    name         'terra' or 'aqua'
    prefix       product-name prefix of the sensor alone: MOD (Terra), MYD (Aqua)
    ga, gq, tbga the 500 m / 1 km surface reflectance, the 250 m surface reflectance and the thermal (Band 31) products
    collection   the collection in the file names (C7: '007')
    validTest    the bands that must be in -100..16000 for a day to be usable (with Band 31 > 0)
    band6ValidDaysOnly  Band 6 is left out of the validity test and composited only from the selected days where it is
                 valid (Aqua: Band 6 is fill on about half the pixels, dead detectors; lead 2026-11-17, rule (i))
    """
    name: str
    prefix: str
    ga: str
    gq: str
    tbga: str
    collection: str = '007'
    validTest: tuple = ('Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_6', 'Band_7')
    band6ValidDaysOnly: bool = False
    # HDF5 groups of the MODIS L2G daily products (C7; confirmed on h09v05 2019065).
    group500m: str = 'HDFEOS/GRIDS/MODIS_Grid_500m_2D/Data Fields/'
    group1km: str = 'HDFEOS/GRIDS/MODIS_Grid_1km_2D/Data Fields/'
    group250m: str = 'HDFEOS/GRIDS/MODIS_Grid_2D/Data Fields/'
    groupThermal: str = 'HDFEOS/GRIDS/MODIS_Grid_2D/Data Fields/'

    @property
    def products(self) -> tuple:
        """(GA, GQ, TBGA) product names, in the order the compositor opens them."""
        return (self.ga, self.gq, self.tbga)


# Band 5 is never tested (Terra band 5 detectors); Terra uses bands 1-4, 6, 7.
TERRA = Sensor(name='terra', prefix='MOD', ga='MOD09GA', gq='MOD09GQ', tbga='MODTBGA')

AQUA = Sensor(name='aqua', prefix='MYD', ga='MYD09GA', gq='MYD09GQ', tbga='MYDTBGA',
              validTest=('Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_7'), band6ValidDaysOnly=True)

SENSORS = {s.name: s for s in (TERRA, AQUA)}


def get(name: str) -> Sensor:
    """The description of a sensor by name ('terra', 'aqua')."""
    if name not in SENSORS:
        raise ValueError('Unknown sensor ' + repr(name) + '; known: ' + ', '.join(SENSORS))
    return SENSORS[name]
