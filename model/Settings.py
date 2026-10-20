import os
from pathlib import Path


# ----------------------------------------------------------------------------
# Settings
#
# Input locations, in one place.  Each is read from an environment variable;
# when it is not set, the NCCS ADAPT location is used.  Command-line options
# (e.g. vcfPredict.py --mod44Dir) still override these.
#
#   VCF_MOD44_DIR    MOD44CH / MOD44CQ 16-day composites: <dir>/<year>/*.hdf
#   VCF_MOD44W_DIR   MOD44W land/water mask (Collection 6.1): <dir>/<year>/001/*.hdf
#   VCF_MOD09A1_DIR  MOD09A1 (optional input, testing only): <dir>/<year>/*.hdf
# ----------------------------------------------------------------------------
MOD44_DIR = Path(os.environ.get('VCF_MOD44_DIR', '/css/modis/Collection6.1/L3/MOD44B-VCF/dev'))
MOD44W_DIR = Path(os.environ.get('VCF_MOD44W_DIR', '/css/modis/Collection6.1/L3/MOD44W-LandWaterMask'))
MOD09A1_DIR = Path(os.environ.get('VCF_MOD09A1_DIR', '/explore/nobackup/projects/ilab/data/MODIS/MOD09A1'))
