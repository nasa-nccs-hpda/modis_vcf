import os
from pathlib import Path


# ----------------------------------------------------------------------------
# Settings
#
# Input locations and collections, in one place, read from environment
# variables.  There are no default folders: a run that reads an input whose
# folder is not set stops with an error naming the variable.  Command-line
# options (e.g. vcfPredict.py --modisDir) override the variables.
#
#   VCF_MOD44_DIR    MOD44CH / MOD44CQ 16-day composites: <dir>/<year>/
#   VCF_MOD44_COLLECTION  collection of those composites: 7 (default,
#                    HDF-EOS5 *.h5) or 61 (HDF4 *.hdf)
#   VCF_MOD44W_DIR   MOD44W land/water mask (Collection 6.1): <dir>/<year>/001/*.hdf
#   VCF_OUTPUT_COLLECTION  collection number in the .h5 product file name and
#                    GranuleID, three digits: 071 (default, this code's
#                    product) or e.g. 007 for an operational MODAPS run
# ----------------------------------------------------------------------------
def _folder(variable: str) -> Path:

    value = os.environ.get(variable, '')
    return Path(value) if value else None


MOD44_DIR = _folder('VCF_MOD44_DIR')
MOD44_COLLECTION = os.environ.get('VCF_MOD44_COLLECTION', '7')
MOD44W_DIR = _folder('VCF_MOD44W_DIR')
OUTPUT_COLLECTION = os.environ.get('VCF_OUTPUT_COLLECTION', '071')
