#!/usr/bin/python

import argparse
from pathlib import Path
import sys

from pyarrow.parquet import ParquetDataset
from pyarrow.parquet import ParquetSchema

from modis_vcf.model.BuildTraining import BuildTraining


# -----------------------------------------------------------------------------
# main
# modis_vcf/view/trainingStatus.py -i /explore/nobackup/projects/ilab/projects/MODIS-VCF/processedTiles -y 2019
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to build training data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-i',
                        type=Path,
                        help='Input directory for finding Parquet files')

    parser.add_argument('-y',
                        type=int,
                        help='The year to check')
    
    args = parser.parse_args()

    # Ensure there is a Parquet file for each training file.
    parqs = list(args.i.glob('*-' + str(args.y) + '.parq'))
    tFiles = list(BuildTraining.TRAINING_DIR.glob('*.samp.bin'))
    numParqs = len(parqs)
    numTFiles = len(tFiles)
    
    print('Training files:', numTFiles)
    print('Parquet files:', numParqs)
    
    if numTFiles != numParqs:
        
        tFileTids = set([t.name.split('.')[0] for t in tFiles])
        parqTids = [p.name.split('-')[0] for p in parqs]
        print('Missing', tFileTids.difference(parqTids))

    # ---
    # Check each Parquet file to ensure they have columns for each metric.
    # Some bands are directly represented in some metrics, while some metrics
    # are statistical composites of bands.  Some metrics also represent bands
    # by day.  Instead of trying to have Metrics tell us all this and as this
    # is meant to be a simple, helpful utility, just encode it all here. 
    # As mentioned elsewhere, there should be a Metric base class with a
    # derivative for each metric.
    # ---
    print('Checking Parquet files for missing metrics.')
    
    expMets = {'BandReflMax': 8,
               'BandReflMedian': 8,
               'BandReflMin': 8,
               'BandReflMaxGreenness': 8,
               'BandReflMedianGreenness': 8,
               'BandReflMinGreenness': 8,
               'BandReflMaxTemp': 8,
               'BandReflMedianTemp': 8,
               'BandReflMinTemp': 8,
               'Lowest3MeanBandRefl': 7,
               'Lowest6MeanBandRefl': 7,
               'Lowest8MeanBandRefl': 7,
               'Greenest3MeanBandRefl': 8,
               'Greenest6MeanBandRefl': 8,
               'Greenest8MeanBandRefl': 8,
               'Warmest3MeanBandRefl': 8,
               'Warmest6MeanBandRefl': 8,
               'Warmest8MeanBandRefl': 8,
               'AmpGreenestBandRefl': 8, 
               'AmpWarmestBandRefl': 8,
               'TempMeanWarmest3': 1,
               'TempMeanGreenest3': 1,
               'UnsortedMonthlyBands': 96}
              
    totalMets = sum(expMets.values()) 
    
    for parq in parqs:
        
        pDs = ParquetDataset(parq)
        cols = pDs.schema.names

        if len(cols) != totalMets + 4:
            print('Missing metrics: ' + str(parq))


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
    