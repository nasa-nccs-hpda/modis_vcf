
import logging
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# BuildTraining
#
# Include all tiles for the given year.
# ----------------------------------------------------------------------------
def BuildTraining(object):
    
    TRAINING_DIR = Path('/explore/nobackup/projects/ilab/data/' + \
                        'MODIS/MODIS_VCF/Mark_training/' + /
                        'VCF_training_adjusted/tile_adjustment/v5.0.3samp/')
    
    TRAINING_DTYPE = np.int16
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, 
                 year: int, 
                 modisDir: Path,
                 metricsDir: Path, 
                 logger: logging.RootLogger):
    
        # Read the samples file.
        samples = np.fromfile(TRAINING_DIR, TRAINING_DTYPE)

        # Get all the metrics for this tile.
        # TODO: make MakeMetrics read existing metrics
        # TODO: change MM.run() to MM.getMetric()
        
        # Get all tile ids.
        # Get the metrics for all tiles.
        
        metrics = Metrics(tileId, year, modisDir, metricsDir, logger)
        
        for metricName
        
        
        
    
    