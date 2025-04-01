# Reproducibility of WEKA models

## Need some local dependencies to deal with this

In the ilab environment:

```bash
pip install liac-arff
```

The models are located here:

```bash
/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/PGE61/pge61/BARE
/explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/PGE61/pge61/BARE/weka.bare.00.model
```

## Get Weka Container

```bash
singularity pull docker://jamesmstone/weka
/explore/nobackup/projects/ilab/containers/weka_latest.sif
```

Another one with the old java version and weka:

```bash
singularity build --sandbox /lscratch/jacaraba/container/weka-test-3.6.2-java5 docker://nasanccs/weka:3.6.2-java5
singularity shell -B $NOBACKUP,/explore/nobackup/projects,/explore/nobackup/people  /lscratch/jacaraba/container/weka-test-3.6.2-java5
```

Running the container:

```bash
singularity shell -B $NOBACKUP,/explore/nobackup/projects,/explore/nobackup/people /explore/nobackup/projects/ilab/containers/weka_latest.sif
```

## Convert parquet files to arff

Activate the ilab anaconda environment, then:

(first argument is the input filename, second argument is the output filename)

Note that this is taking one of the parquet files used for prediction, to ingest one of the ones from training,
you might need to do some modifications to the columns coming in.

```bash
python 1_parquet_to_arff.py '/explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics.parquet' 'my_test.arff'
```

Subset filename:

```bash
/explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics_subset.arff
```

## Run models against arff

Reading the model

```bash
java -cp /weka/weka.jar weka.classifiers.trees.M5P -l /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/PGE61/pge61/BARE/weka.bare.00.model -p 0
```

Another example:

```bash
for chunk in /explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics_Y_chunk_*.arff; do singularity exec -B $NOBACKUP,/explore/nobackup/projects,/explore/nobackup/people /explore/nobackup/projects/ilab/containers/weka-test-3.6.2-java5 java -Xmx64g -cp /opt/weka-3-6-2/weka.jar weka.classifiers.trees.M5P -l /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/PGE61/pge61/BARE/weka.bare.01.model -T "$chunk" -p 0 >> /explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics_Y_predictions_all_model01.txt; done
```

A bigger example:

```bash
java -cp /opt/weka-3-6-2/weka.jar weka.classifiers.trees.M5P -l /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/PGE61/pge61/BARE/weka.bare.00.model -T /explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics_Class.arff -p 0 > /explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/prediction_MOD44C_h20v06_2019_3-Metrics_Class.txt
```

## Convert outputs to geotiff

Getting the features

```bash
mkdir -p /explore/nobackup/people/jacaraba/ForPeople/ForMark/VCF/MOD44B_VCF_Development/PGE61/pge61/BARE && \
for model in /explore/nobackup/people/mcarrol2/MOD44B_VCF_Development/PGE61/pge61/BARE/*.model; do strings "$model" | grep -E "^n[0-9]{3}" | sort -u | sed 's/psq$//' > "/explore/nobackup/people/jacaraba/ForPeople/ForMark/VCF/MOD44B_VCF_Development/PGE61/pge61/BARE/$(basename "$model" .model)_strings.txt"; done
```

## Translating n* features to current features from dataframe

```bash
n014 == UnsortedMonthlyBands-Band_2-Day-2019225
n018 == UnsortedMonthlyBands-Band_3-Day-2019065
n053 == UnsortedMonthlyBands-Band_6-Day-2020017
n056 == UnsortedMonthlyBands-Band_7-Day-2019097
n060 == UnsortedMonthlyBands-Band_7-Day-2019321
n061 == UnsortedMonthlyBands-Band_7-Day-2019353
n062 == UnsortedMonthlyBands-Band_7-Day-2020017
n063 == UnsortedMonthlyBands-NDVI-Day-2019065
n064 == UnsortedMonthlyBands-NDVI-Day-2019097
n065 == UnsortedMonthlyBands-NDVI-Day-2019129
n066 == UnsortedMonthlyBands-NDVI-Day-2019193
n067 == UnsortedMonthlyBands-NDVI-Day-2019225
n069 == UnsortedMonthlyBands-NDVI-Day-2019321
n070 == UnsortedMonthlyBands-NDVI-Day-2019353
n071 == UnsortedMonthlyBands-NDVI-Day-2020017
n086 == Lowest6MeanBandRefl-Band_6
n109 == BandReflMedian-Band_2
n111 == BandReflMedian-Band_4
n140 == BandReflMedian-Band_2
n189 == BandReflMin-NDVI
n223 == TempMeanGreenest3
```

-no-class
-p 0
-classifications weka.classifiers.evaluation.output.prediction.CSV > predictions.csv
