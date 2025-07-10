#!/usr/bin/python

import argparse
import glob
import logging
from pathlib import Path
import sys

from modis_vcf.model.BuildTraining import BuildTraining
from modis_vcf.model.BuildTraining import TrainingType

# -----------------------------------------------------------------------------
# main
#
# python modis_vcf/view/buildTraining.py -y 2019 --metricsDir /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/SystemTests/1-Metrics -o /explore/nobackup/people/rlgill/SystemTesting/modis-vcf/SystemTests/2-Training -t h09v05 h11v02 --trainingType pcttree   # noqa: E501
#
# h08v04 h08v05 h09v04 h09v05 h10v04 h10v05 h10v06 h11v02 h11v03 h11v04 h11v05 h11v08 h11v09 h11v10 h12v01 h12v02 h12v03 h12v04 h12v05 h12v09 h12v10 h12v12 h13v01 h13v02 h13v10 h13v11 h13v12 h16v01 h17v05 h18v03 h18v04 h18v07 h19v04 h19v07 h19v08 h19v09 h19v10 h19v11 h19v12 h20v02 h20v03 h20v04 h20v06 h20v08 h20v09 h20v10 h20v11 h21v01 h21v02 h21v04 h21v05 h21v06 h21v10 h22v03 h22v04 h23v02 h23v03 h24v02 h24v03 h24v04 h26v06 h27v04 h27v06 h27v07 h28v11 h29v11 h29v12 h30v12 h31v11   # noqa: E501
# -----------------------------------------------------------------------------
def main():
    
    desc = 'Use this application to build training data.'
    parser = argparse.ArgumentParser(description=desc)

    modisDir = Path('/css/modis/Collection6.1/L3/MOD44B-VCF/dev')

    parser.add_argument('--metricsDir',
                        type=Path,
                        required='True',
                        help='Directory for reading or writing metrics')

    parser.add_argument('--modisDir',
                        type=Path,
                        default=modisDir,
                        help='Input directory for finding MODIS files')

    parser.add_argument('-m',
                        type=str,
                        nargs='*',
                        help='A space-separated list of metric names')

    parser.add_argument('-o',
                        type=Path,
                        default='.',
                        help='Output directory for writing training files')

    parser.add_argument('-t',
                        type=str,
                        nargs='*',
                        help='A space-separated list of tile IDs in ' +
                             'the form h##v## h##v##')

    parser.add_argument('--trainingDir',
                        type=Path,
                        help='Input directory for finding training files')

    parser.add_argument('--trainingType',
                        type=TrainingType,
                        choices=[t.value for t in TrainingType],
                        help='Choose the training type to run.')

    parser.add_argument('-y',
                        type=int,
                        required='True',
                        help='The year to run')

    args = parser.parse_args()
    
    # Get the tile IDs of the sample data.
    year = args.y
    modisDir = args.modisDir
    metricsDir = args.metricsDir
    outDir = args.o
    trainingDir = args.trainingDir
    trainingType = args.trainingType
    tids = args.t
    metricNames = args.m
    
    bt = BuildTraining(year=year, 
                       modisDir=modisDir,
                       metricsDir=metricsDir, 
                       outDir=outDir, 
                       trainingDir=trainingDir,
                       trainingType=trainingType, 
                       tileIds=tids, 
                       metricNames=metricNames)

    bt.run()
        

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
