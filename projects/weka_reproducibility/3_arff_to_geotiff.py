import re
import os
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
from glob import glob
from pathlib import Path

# Input file path
input_file = '/explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics_Y_chunk_*.arff_prediction.txt'
filenames = sorted(glob(input_file))

# Regular expression to match the prediction lines
pattern = r'^\s*(\d+)\s+(\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s*$'

# List to store predictions
predictions = []

# Read the input file and extract predictions
for i in range(1, 25):
    """
    with open(input_file, 'r') as infile:
        for line in infile:
            match = re.match(pattern, line)
            if match:
                instance, actual, predicted, error = match.groups()
                predictions.append(float(predicted))
    """
    input_file = f'/explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics_Y_chunk_{i}.arff_prediction.txt'
    print(input_file)

    with open(input_file, 'r') as infile:
        lines = infile.readlines()[5:]
        for line in lines:
            filtered = line.rstrip('\n').split()
            if len(filtered) == 4:
                predictions.append(float(filtered[2]))
        #for line in infile:
        #    match = re.match(pattern, line)
        #    if match:
        #        instance, actual, predicted, error = match.groups()
        #       predictions.append(float(predicted))

    print(len(predictions))    

# If you need to save the DataFrame for later use
# df.to_parquet('/path/to/save/predictions.parquet')

# If you need to save the numpy array
# np.save('/path/to/save/predictions.npy', predictions_array)


def save_output_raster(raster_data_path, prediction, output_dir):
    
    # get one of the rasters for reference
    image = rxr.open_rasterio(raster_data_path)
    image = image.drop(
        dim="band",
        labels=image.coords["band"].values[1:],
    )
    output_filename = os.path.join(output_dir, f'{"_".join(Path(raster_data_path).parts[-5:-1])}_weka.tif')

    # save prediction in raster
    prediction = xr.DataArray(
        np.expand_dims(prediction, axis=0),
        name='',
        coords=image.coords,
        dims=image.dims,
        attrs=image.attrs
    )
    # Add metadata to raster attributes
    prediction.attrs['long_name'] = (output_filename)

    # Set nodata values on mask
    nodata = prediction.rio.nodata

    prediction = prediction.where(image != nodata)
    nodata=250
    prediction.rio.write_nodata(nodata, encoded=True, inplace=True)

    # Save output raster file to disk
    prediction.rio.to_raster(
        output_filename,
        BIGTIFF="IF_SAFER",
        compress='LZW',
        driver='GTiff',
        dtype='float32'
    )
    return

# Take the predicted and put it into a geotiff
predictions_array = np.array(predictions)
print(np.unique(predictions_array, return_counts=True))


predictions_reshaped = np.round(predictions_array.reshape(4800, 4800))


save_output_raster(
    '/explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics.tif',
    predictions_reshaped,
    '/explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test'
)

