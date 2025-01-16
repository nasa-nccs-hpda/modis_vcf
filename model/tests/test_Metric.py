
from pathlib import Path
import unittest

import numpy as np

from modis_vcf.model.Metric import Metric
from modis_vcf.model.Metric import NewBand
from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44


# -----------------------------------------------------------------------------
# class MetricTestCase
#
# python -m unittest discover modis_vcf/model/tests/
# python -m unittest modis_vcf.model.tests.test_Metric
# python -m unittest modis_vcf.model.tests.test_Metric.MetricTestCase.testInit
# -----------------------------------------------------------------------------
class MetricTestCase(unittest.TestCase):

    # -------------------------------------------------------------------------
    # setUp
    # -------------------------------------------------------------------------
    def setUp(self):

        self._mod44InDir = \
            Path('/explore/nobackup/projects/ilab/data/MODIS/MOD44C')

        self.metricName = 'Warmest3MeanBandRefl'
        self.band = NewBand.BAND1
        self.pt44 = ProductTypeMod44(self._mod44InDir)
        self.tid = 'h09v05'
        self.year = 2019

        self.outDir = Path('/explore/nobackup/people/rlgill' + 
                           '/SystemTesting/modis-vcf/MOD44') 

    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        # ---
        # Test finding metrics.  This goes away when metric subclasses are
        # implemented.
        # ---
        mName = 'metricWarmest3MeanBandRefl'
        
        m = Metric(name=mName,
                   # band=self.band,
                   tileId=self.tid, 
                   year=self.year, 
                   productType=self.pt44, 
                   outDir=self.outDir)
                   
        self.assertEqual(m.name, mName)
        self.assertEqual(m._productType, self.pt44)

        m = Metric(name=self.metricName,
                   # band=self.band,
                   tileId=self.tid, 
                   year=self.year, 
                   productType=self.pt44, 
                   outDir=self.outDir)
                   
        self.assertEqual(m.name, 'metric' + self.metricName)

        with self.assertRaisesRegex(RuntimeError, 'Unknown metric'):

            m = Metric(name='invalidMetric',
                       # band=self.band,
                       tileId=self.tid, 
                       year=self.year, 
                       productType=self.pt44, 
                       outDir=self.outDir)

        # Test product type
        with self.assertRaisesRegex(RuntimeError, 'product type must be'):

            m = Metric(name=self.metricName,
                       # band=self.band,
                       tileId=self.tid, 
                       year=self.year, 
                       productType=None, 
                       outDir=self.outDir)
                   
        # Test outDir.
        with self.assertRaisesRegex(RuntimeError, 'valid output directory'):

            m = Metric(name=self.metricName,
                       # band=self.band,
                       tileId=self.tid, 
                       year=self.year, 
                       productType=None, 
                       outDir=None)
                   
        with self.assertRaisesRegex(RuntimeError, 'valid output directory'):

            m = Metric(name=self.metricName,
                       # band=self.band,
                       tileId=self.tid, 
                       year=self.year, 
                       productType=None, 
                       outDir=Path('does/not/exist'))

        # Test band.
        with self.assertRaisesRegex(RuntimeError, 'Invalid band'):

            m = Metric(name=self.metricName,
                       # band='junk',
                       tileId=self.tid, 
                       year=self.year, 
                       productType=self.pt44, 
                       outDir=self.outDir)
                   
        with self.assertRaisesRegex(RuntimeError, 'Invalid band'):

            m = Metric(name=self.metricName,
                       # band=None,
                       tileId=self.tid, 
                       year=self.year, 
                       productType=self.pt44, 
                       outDir=self.outDir)
                   
    # -------------------------------------------------------------------------
    # testComputeMyMetric
    # -------------------------------------------------------------------------
    def testComputeMyMetric(self):

        m = Metric(name=self.metricName,
                   # band=self.band,
                   tileId=self.tid,
                   year=self.year,
                   productType=self.pt44,
                   outDir=self.outDir)

        value: nd.array = m._myComputeMetric()
        self.assertEqual(value.shape, (4800, 4800))
        self.assertEqual(value.dtype, np.int64)
        
        m = Metric(name='UnsortedMonthlyBands',
                   # band=self.band,
                   tileId=self.tid,
                   year=self.year,
                   productType=self.pt44,
                   outDir=self.outDir)

        import pdb
        pdb.set_trace()
        value: nd.array = m._myComputeMetric()
        self.assertEqual(value.shape, (11, 4800, 4800))
        
    # -------------------------------------------------------------------------
    # testSetTid
    # -------------------------------------------------------------------------
    def testSetTid(self):
        
        tid = 'h09v05'
        
        m = Metric(name=self.metricName,
                   # band=self.band,
                   tileId=tid, 
                   year=self.year, 
                   productType=self.pt44, 
                   outDir=self.outDir)
                   
        self.assertEqual(m._tid, tid)
        
        tid = 'h00v00'
        
        m = Metric(name=self.metricName,
                   # band=self.band,
                   tileId=tid, 
                   year=self.year, 
                   productType=self.pt44, 
                   outDir=self.outDir)
                   

        self.assertEqual(m._tid, tid)
        
        tid = 'h17v35'
        
        m = Metric(name=self.metricName,
                   # band=self.band,
                   tileId=tid, 
                   year=self.year, 
                   productType=self.pt44, 
                   outDir=self.outDir)
        
        self.assertEqual(m._tid, tid)
        
        with self.assertRaisesRegex(ValueError, 'Invalid tile ID'):

            tid = 'h17v36'

            m = Metric(name=self.metricName,
                       # band=self.band,
                       tileId=tid, 
                       year=self.year, 
                       productType=self.pt44, 
                       outDir=self.outDir)
        
        with self.assertRaisesRegex(ValueError, 'Invalid tile ID'):

            tid = 'h18v35'

            m = Metric(name=self.metricName,
                       # band=self.band,
                       tileId=tid, 
                       year=self.year, 
                       productType=self.pt44, 
                       outDir=self.outDir)
        
    # -------------------------------------------------------------------------
    # testSetYear
    # -------------------------------------------------------------------------
    def testSetYear(self):
        
        m = Metric(name=self.metricName,
                   # band=self.band,
                   tileId=self.tid, 
                   year=self.year, 
                   productType=self.pt44, 
                   outDir=self.outDir)
                   
        self.assertEqual(m._year, self.year)
        
        with self.assertRaisesRegex(ValueError, 'Invalid year'):

            year = 1989
            
            m = Metric(name=self.metricName,
                       # band=self.band,
                       tileId=self.tid, 
                       year=year, 
                       productType=self.pt44, 
                       outDir=self.outDir)

        with self.assertRaisesRegex(ValueError, 'Invalid year'):

            year = 4000

            m = Metric(name=self.metricName,
                       # band=self.band,
                       tileId=self.tid, 
                       year=year, 
                       productType=self.pt44, 
                       outDir=self.outDir)

