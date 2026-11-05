"""
The Collection 7 MOD44B product file: one HDF5 file per tile and year, in the HDF-EOS5 layout of the MODAPS C7 granules
(LAADS allData/7/MOD44B, made with the C code and API-Trans; template MOD44B.A2019065.h08v05.007.2026292174253.h5).

Same groups, fields, types, chunks, filters, attributes and root metadata as those granules, with three corrections
(docs/BUILD.md):
  - Percent_Tree_Cover_SD / Percent_NonVegetated_SD valid_range 0-5000 (SD x 100 of 0-100 values; the C code says
    0-275, which its own values exceed);
  - out-of-projection pixels keep 250 and the percent layers declare it (OutOfProjection_Value), where the C code
    predicts on the fill inputs and flags nothing;
  - the run-specific metadata (versions, host, inputs) are this code's own; MODAPS can override any root attribute.

The layer values are those of the GeoTIFFs (VcfPredict._write), unchanged.
"""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import platform

import h5py
import numpy as np


CODE_VERSION = '1.7.1'
COLLECTION = '007'

# Root attributes of a Collection 7 MOD44CQ input that give the tile bounds (degrees).
BOUND_ATTRS = dict(north='geospatial_lat_max', south='geospatial_lat_min', east='geospatial_lon_max',
                   west='geospatial_lon_min')

GRID = 'MOD44B_250m_GRID'
GRID_PATH = '/HDFEOS/GRIDS/' + GRID
FIELDS_PATH = GRID_PATH + '/Data Fields'
SIZE = 4800
EARTH_RADIUS = 6371007.181

WATER = 200
OUT_OF_PROJ = 250
PCT_FILL = 253

# The 7 fields, in the C code's order (make_grid.c): name, dtype, long_name, units, valid_range, _FillValue, extras.
_PCT = dict(units='Percent Fractional Cover', valid=(0, 100), fill=PCT_FILL,
            extra=dict(Water_Value=WATER, OutOfProjection_Value=OUT_OF_PROJ))
_SD = dict(units='Percent Fractional Cover', valid=(0, 5000), fill=0, extra=dict(scale_factor=np.float64(0.01)))
_FLAGS = dict(units='none', valid=(0, 255), fill=0, extra={})

FIELDS = [('Percent_Tree_Cover', np.uint8, 'Percent Tree Cover', _PCT),
          ('Percent_NonTree_Vegetation', np.uint8, 'Percent Nontree Vegetation Cover', _PCT),
          ('Percent_NonVegetated', np.uint8, 'Percent Nonvegetated Cover', _PCT),
          ('Quality', np.uint8, 'Quality Flags', _FLAGS),
          ('Percent_Tree_Cover_SD', np.uint16, 'Standard Deviation of Bagged Models, Percent Tree Cover', _SD),
          ('Percent_NonVegetated_SD', np.uint16, 'Standard Deviation of Bagged Models, Percent Nonvegetated Cover',
           _SD),
          ('Cloud', np.uint8, 'Cloud Cover in Composites', _FLAGS)]

FIELD_NAMES = [f[0] for f in FIELDS]

LONG_NAME = 'MODIS/Terra Vegetation Continuous Fields Yearly L3 Global 250m SIN Grid'
PLACEHOLDER = 'set Passed as default, will decide later'

# Root attributes that do not depend on the tile, as in the C7 granules.  Everything here can be overridden.
FIXED_ATTRS = {
    'AssociatedSensorShortName': 'MODIS',
    'AutomaticQualityFlag': 'Passed',
    'AutomaticQualityFlagExplanation': PLACEHOLDER,
    'Conventions': 'CF-1.6, ACDD-1.3',
    'DESCRrevision': '7',
    'ExclusionGRingFlag': 'N',
    'LocalVersionID': 'modis_vcf ' + CODE_VERSION,
    'LongName': LONG_NAME,
    'OperationalQualityFlag': 'Passed',
    'OperationalQualityFlagExplanation': PLACEHOLDER,
    'PGEVersion': CODE_VERSION,
    'ParameterName': LONG_NAME,
    'ProcessingCenter': 'MODAPS',
    'QAPERCENTGOODQUALITY': '100',
    'QAPERCENTNOTPRODUCEDCLOUD': '0',
    'QAPERCENTNOTPRODUCEDOTHER': '0',
    'QAPERCENTOTHERQUALITY': '100',
    'QAPercentCloudCover': np.array([0], np.int32),
    'QAPercentInterpolatedData': np.array([0], np.int32),
    'QAPercentMissingData': np.array([0], np.int32),
    'QAPercentOutofBoundsData': np.array([0], np.int32),
    'ReprocessingActual': 'reprocessed',
    'ReprocessingPlanned': 'further update is anticipated',
    'ScienceQualityFlag': 'Not Investigated',
    'ScienceQualityFlagExplanation': 'See https://landweb.modaps.eosdis.nasa.gov/prodqa?sat=Terra for the product '
                                     'Science Quality status.',
    'ShortName': 'MOD44B',
    'VersionID': np.array([7], np.int32),
    '_AdditionalAttributeNames': 'QAPERCENTGOODQUALITY,QAPERCENTOTHERQUALITY,QAPERCENTNOTPRODUCEDCLOUD,'
                                 'QAPERCENTNOTPRODUCEDOTHER,HORIZONTALTILENUMBER,VERTICALTILENUMBER,TileID,'
                                 'identifier_product_doi,identifier_product_doi_authority',
    'creator_email': 'modis-ops@lists.nasa.gov',
    'creator_name': 'MODAPS',
    'creator_url': 'https://ladsweb.modaps.eosdis.nasa.gov',
    'identifier_product_doi': '10.5067/MODIS/MOD44B.' + COLLECTION,
    'institution': 'NASA Goddard Space Flight Center, LAADS',
    'instrument': 'MODIS',
    'keywords_vocabulary': 'NASA Global Change Master Directory (GCMD) Science Keywords',
    'license': 'https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance',
    'naming_authority': 'https://doi.org',
    'platform': 'Terra',
    'publisher_email': 'modis-ops@lists.nasa.gov',
    'publisher_name': 'LAADS',
    'publisher_url': 'https://ladsweb.modaps.eosdis.nasa.gov',
}


# ----------------------------------------------------------------------------
# granuleName
# ----------------------------------------------------------------------------
def granuleName(tid: str, year: int, production: datetime) -> str:
    """MOD44B.A<yyyy>065.<tile>.007.<yyyydddhhmmss>.h5, the C7 file name."""
    return 'MOD44B.A%d065.%s.%s.%s.h5' % (year, tid, COLLECTION, production.strftime('%Y%j%H%M%S'))


# ----------------------------------------------------------------------------
# tile helpers
# ----------------------------------------------------------------------------
def _hv(tid: str) -> tuple:
    return int(tid[tid.index('h') + 1:tid.index('v')]), int(tid[tid.index('v') + 1:])


def _corners(tid: str) -> tuple:
    """Upper left and lower right of the tile in metres (as VcfPredict._geoTransform)."""
    from modis_vcf.model.VcfPredict import VcfPredict
    h, v = _hv(tid)
    size = VcfPredict.MODIS_TILE_SIZE
    ulx = VcfPredict.MODIS_GRID_ULX + h * size
    uly = VcfPredict.MODIS_GRID_ULY - v * size
    return ulx, uly, ulx + size, uly - size


def _geoFromCorners(tid: str) -> dict:
    """Bounds and G-ring from the tile corners, for when no input metadata is at hand (rule tiles)."""
    ulx, uly, lrx, lry = _corners(tid)
    pts = [(ulx, lry), (ulx, uly), (lrx, uly), (lrx, lry)]           # the C7 order: LL, UL, UR, LR
    lat = [y / EARTH_RADIUS for _, y in pts]
    lon = [np.clip(x / (EARTH_RADIUS * np.cos(la)), -np.pi, np.pi) for (x, _), la in zip(pts, lat)]
    lat, lon = np.degrees(lat), np.degrees(lon)
    return dict(GRingPointLatitude=lat, GRingPointLongitude=lon, north=lat.max(), south=lat.min(),
                west=lon.min(), east=lon.max(), DayNightFlag='Day')


# ----------------------------------------------------------------------------
# modlandBounds
# ----------------------------------------------------------------------------
def modlandBounds(tid: str) -> tuple:
    """West, east, south and north of a tile from the MODLAND table sn_bound_10deg.txt (modis-land.gsfc.nasa.gov/pdf, 4 decimals;
    stored here with LF line ends).  The Collection 6.1 MOD44C bounding coordinates equal it on all 292 tiles (within 5e-5)."""

    h, v = _hv(tid)

    for line in (Path(__file__).parent / 'sn_bound_10deg.txt').read_text().splitlines():

        f = line.split()

        if len(f) == 6 and f[0].isdigit() and (int(f[0]), int(f[1])) == (v, h):

            if float(f[2]) == -999:
                raise ValueError('No MODLAND bounds for fill tile ' + tid)

            return tuple(float(x) for x in f[2:])

    raise ValueError('Tile not in the MODLAND table: ' + tid)


# ----------------------------------------------------------------------------
# inputGeo
# ----------------------------------------------------------------------------
def inputGeo(inputFile: Path) -> dict:
    """G-ring, bounds and day/night flag of an input composite (MOD44CQ), as the C code copies them.  Collection 7 (.h5)
    inputs hold the G-ring (space separated) and the bounds as the root attributes geospatial_lat_max / _lat_min /
    _lon_max / _lon_min; the MODLAND table is used only when an input lacks them."""

    from osgeo import gdal
    md = gdal.Open(str(inputFile)).GetMetadata()

    if Path(inputFile).suffix == '.h5':

        with h5py.File(inputFile, 'r') as f:
            b = {k: float(np.ravel(f.attrs[a])[0]) for k, a in BOUND_ATTRS.items() if a in f.attrs}

        if len(b) == 4:
            west, east, south, north = b['west'], b['east'], b['south'], b['north']
        else:
            west, east, south, north = modlandBounds(Path(inputFile).name.split('.')[2])

        return dict(GRingPointLatitude=np.array([float(x) for x in md['GRingPointLatitude'].split()]),
                    GRingPointLongitude=np.array([float(x) for x in md['GRingPointLongitude'].split()]),
                    north=north, south=south, west=west, east=east, DayNightFlag=md.get('DayNightFlag', 'Day'))

    nums = lambda k: np.array([float(x) for x in md[k].split(',')])
    return dict(GRingPointLatitude=nums('GRINGPOINTLATITUDE.1'), GRingPointLongitude=nums('GRINGPOINTLONGITUDE.1'),
                north=float(md['NORTHBOUNDINGCOORDINATE']), south=float(md['SOUTHBOUNDINGCOORDINATE']),
                west=float(md['WESTBOUNDINGCOORDINATE']), east=float(md['EASTBOUNDINGCOORDINATE']),
                DayNightFlag=md.get('DAYNIGHTFLAG', 'Day'))


# ----------------------------------------------------------------------------
# structMetadata
# ----------------------------------------------------------------------------
def structMetadata(tid: str) -> str:
    """StructMetadata.0 as API-Trans writes it for this grid."""

    ulx, uly, lrx, lry = _corners(tid)
    dims = [('XDim', SIZE), ('YDim', SIZE), ('dimofone', 1)]
    types = {np.uint8: 'H5T_NATIVE_UCHAR', np.uint16: 'H5T_NATIVE_USHORT'}
    t = ['GROUP=SwathStructure', 'END_GROUP=SwathStructure', 'GROUP=GridStructure', '\tGROUP=GRID_1',
         '\t\tGridName="%s"' % GRID, '\t\tXDim=%d' % SIZE, '\t\tYDim=%d' % SIZE,
         '\t\tUpperLeftPointMtrs=(%.6f,%.6f)' % (ulx, uly), '\t\tLowerRightMtrs=(%.6f,%.6f)' % (lrx, lry),
         '\t\tGridOrigin=HE5_HDFE_GD_UL', '\t\tProjection=HE5_GCTP_SNSOID',
         '\t\tProjParams=(%.6f,0,0,0,0,0,0,0,0,0,0,0,0)' % EARTH_RADIUS, '\t\tSphereCode=-1', '\t\tGROUP=Dimension']

    for i, (name, n) in enumerate(dims, 1):
        t += ['\t\t\tOBJECT=Dimension_%d' % i, '\t\t\t\tDimensionName="%s"' % name, '\t\t\t\tSize=%d' % n,
              '\t\t\tEND_OBJECT=Dimension_%d' % i]

    t += ['\t\tEND_GROUP=Dimension', '\t\tGROUP=DataField']
    fields = [(n, types[d], '("YDim","XDim")') for n, d, _, _ in FIELDS] + \
             [('Projection', 'H5T_NATIVE_INT', '("dimofone")')]

    for i, (name, typ, dl) in enumerate(fields, 1):
        t += ['\t\t\tOBJECT=DataField_%d' % i, '\t\t\t\tDataFieldName="%s"' % name, '\t\t\t\tDataType=%s' % typ,
              '\t\t\t\tDimList=%s' % dl, '\t\t\t\tMaxdimList=%s' % dl, '\t\t\tEND_OBJECT=DataField_%d' % i]

    t += ['\t\tEND_GROUP=DataField', '\t\tGROUP=MergedFields', '\t\tEND_GROUP=MergedFields', '\tEND_GROUP=GRID_1',
          'END_GROUP=GridStructure', 'GROUP=PointStructure', 'END_GROUP=PointStructure', 'GROUP=ZaStructure',
          'END_GROUP=ZaStructure', 'END', '']
    return '\n'.join(t)


# ----------------------------------------------------------------------------
# string attributes and datasets as API-Trans writes them: fixed length, null terminated, ASCII
# ----------------------------------------------------------------------------
def _strType(n: int):
    t = h5py.h5t.C_S1.copy()
    t.set_size(n)
    t.set_strpad(h5py.h5t.STR_NULLTERM)
    return t


def _setStr(obj, name: str, value: str, size: int = None, exact: bool = False) -> None:
    """exact: no room for the terminating null, as the C code's SDsetattr(strlen) strings and API-Trans's DOI."""

    b = value.encode('ascii')
    n = size or len(b) + (0 if exact else 1)

    if name in obj.attrs:
        del obj.attrs[name]

    aid = h5py.h5a.create(obj.id, name.encode(), _strType(n), h5py.h5s.create(h5py.h5s.SCALAR))
    aid.write(np.array(b, dtype='S%d' % n), mtype=_strType(n))


def _setAttr(obj, name: str, value) -> None:

    if isinstance(value, str):
        _setStr(obj, name, value, exact=name in _EXACT_ROOT_STRINGS)
    else:
        obj.attrs.create(name, np.atleast_1d(value))


# Root strings stored without the terminating null in the C7 granules (all others have it).
_EXACT_ROOT_STRINGS = {'identifier_product_doi'}


# ----------------------------------------------------------------------------
# write
# ----------------------------------------------------------------------------
def write(outDir: Path,
          tid: str,
          year: int,
          layers: dict,
          inputs: list = (),
          geo: dict = None,
          overrides: dict = None,
          production: datetime = None) -> Path:
    """
    Write the C7 file of one tile and year and return its path.  layers: {field name: 4800 x 4800 array} for the 7
    fields (values as in the GeoTIFFs; a missing SD layer is written as its fill, 0).  inputs: the input file names
    for InputDataProducts.  geo: inputGeo() of the first MOD44CQ, or None to compute it from the tile corners.
    overrides: root attributes to replace or add (for MODAPS: PGEVersion, ProcessingEnvironment, ...).
    The file is written under a temporary name and renamed when complete.
    """

    production = production or datetime.now(timezone.utc)
    geo = geo or _geoFromCorners(tid)
    h, v = _hv(tid)
    name = granuleName(tid, year, production)
    outPath = Path(outDir) / name
    tmp = outPath.with_name(name + '.part')
    ulx, uly, _, _ = _corners(tid)
    px = (_corners(tid)[2] - ulx) / SIZE

    begin = datetime(year, 1, 1) + timedelta(days=64)
    end = datetime(year + 1, 1, 1) + timedelta(days=64)
    attrs = dict(FIXED_ATTRS)
    attrs.update({
        'DayNightFlag': geo['DayNightFlag'],
        'GRingPointLatitude': np.asarray(geo['GRingPointLatitude'], np.float64),
        'GRingPointLongitude': np.asarray(geo['GRingPointLongitude'], np.float64),
        'GRingPointSequenceNo': np.array([1, 2, 3, 4], np.int32),
        'GranuleID': name,
        'HORIZONTALTILENUMBER': str(h),
        'VERTICALTILENUMBER': str(v),
        'TileID': '51%03d%03d' % (h, v),
        'InputDataProducts': ','.join(Path(str(i)).name for i in inputs),
        'ProcessingEnvironment': ' '.join(platform.uname()),
        'ProductionDateTime': production.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        'RangeBeginningDate': begin.strftime('%Y-%m-%d'),
        'RangeBeginningTime': '00:00:00.000000',
        'RangeEndingDate': end.strftime('%Y-%m-%d'),
        'RangeEndingTime': '00:00:00.000000',
        'geospatial_lat_max': np.float64(geo['north']),
        'geospatial_lat_min': np.float64(geo['south']),
        'geospatial_lon_max': np.float64(geo['east']),
        'geospatial_lon_min': np.float64(geo['west']),
    })
    attrs.update(overrides or {})

    try:
        with h5py.File(tmp, 'w') as f:

            for k in sorted(attrs):
                _setAttr(f, k, attrs[k])

            f.create_group('HDFEOS/ADDITIONAL/FILE_ATTRIBUTES')
            g = f.create_group(GRID_PATH.lstrip('/'))
            fields = g.create_group('Data Fields')
            heInfo = f.create_group('HDFEOS INFORMATION')
            _setStr(heInfo, 'HDFEOSVersion', 'HDFEOS_5.1.16', size=32)
            sm = structMetadata(tid).encode('ascii')
            sid = h5py.h5d.create(heInfo.id, b'StructMetadata.0', _strType(32000), h5py.h5s.create(h5py.h5s.SCALAR))
            sid.write(h5py.h5s.ALL, h5py.h5s.ALL, np.array(sm, dtype='S32000'), mtype=_strType(32000))

            # Dimension scales: pixel edges in metres from the upper left, as in the C7 granules.
            scales = {}

            for dim, axis, start, step in (('XDim', 'x', ulx, px), ('YDim', 'y', uly, -px)):
                d = g.create_dataset(dim, data=start + step * np.arange(SIZE, dtype=np.float64), chunks=(SIZE,),
                                     shuffle=True, compression='gzip', compression_opts=4)
                d.make_scale(axis)
                _setStr(d, 'coverage_content_type', 'coordinate')
                _setStr(d, 'long_name', axis + ' coordinate of projection')
                _setStr(d, 'standard_name', 'projection_' + axis + '_coordinate')
                _setStr(d, 'units', 'm')
                scales[dim] = d

            for fname, dtype, longName, spec in FIELDS:

                a = layers.get(fname)

                if a is None:
                    a = np.full((SIZE, SIZE), spec['fill'], dtype)

                lim = np.iinfo(dtype)

                if a.min() < lim.min or a.max() > lim.max:
                    raise ValueError(fname + ': values do not fit ' + np.dtype(dtype).name)

                d = fields.create_dataset(fname, data=a.astype(dtype), chunks=(SIZE, SIZE), shuffle=True,
                                          compression='gzip', compression_opts=4, fillvalue=dtype(spec['fill']))
                d.dims[0].attach_scale(scales['YDim'])
                d.dims[1].attach_scale(scales['XDim'])
                d.attrs.create('_FillValue', np.array([spec['fill']], dtype))
                _setStr(d, 'coordinates', GRID_PATH + '/YDim ' + GRID_PATH + '/XDim')
                _setStr(d, 'grid_mapping', 'Projection')
                _setStr(d, 'long_name', longName, exact=True)
                _setStr(d, 'units', spec['units'], exact=True)
                d.attrs.create('valid_range', np.array(spec['valid'], dtype))

                for k, val in spec['extra'].items():
                    d.attrs.create(k, np.array([val], val.dtype if isinstance(val, np.generic) else dtype))

            p = fields.create_dataset('Projection', data=np.array([0], np.int32))
            p.attrs.create('earth_radius', np.array([EARTH_RADIUS]))
            p.attrs.create('false_easting', np.array([0.0]))
            p.attrs.create('false_northing', np.array([0.0]))
            _setStr(p, 'grid_mapping_name', 'sinusoidal')
            p.attrs.create('longitude_of_central_meridian', np.array([0.0]))

        os.replace(tmp, outPath)

    finally:
        if tmp.exists():
            tmp.unlink()

    return outPath


# ----------------------------------------------------------------------------
# find / read
# ----------------------------------------------------------------------------
def find(mapsDir: Path, tid: str, year: int) -> Path:
    """The newest C7 file of a tile and year in mapsDir, or None."""
    files = sorted(Path(mapsDir).glob('MOD44B.A%d065.%s.%s.*.h5' % (year, tid, COLLECTION)))
    return files[-1] if files else None


def readLayer(path: Path, field: str) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        return f[FIELDS_PATH + '/' + field][()]


def readMapLayer(mapsDir: Path, tid: str, year: int, field: str) -> np.ndarray:
    """A layer from the GeoTIFF when there is one (<field>-<tile>-<year>.tif), else from the C7 file."""

    tif = Path(mapsDir) / ('%s-%s-%d.tif' % (field, tid, year))

    if tif.exists():
        from osgeo import gdal
        return gdal.Open(str(tif)).ReadAsArray()

    h5 = find(mapsDir, tid, year)

    if h5 is None:
        raise FileNotFoundError('No ' + field + ' map for ' + tid + ' ' + str(year) + ' in ' + str(mapsDir))

    return readLayer(h5, field)
