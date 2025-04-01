import sys
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import geopandas as gpd
import arff
import math

# Load Parquet file
df = pd.read_parquet('/explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics.parquet')
df = df.drop(['x', 'y', 'x_idx', 'y_idx'], axis=1)

column_mapping = {
    'UnsortedMonthlyBands-Band_2-Day-2019225': 'n014',
    'UnsortedMonthlyBands-Band_3-Day-2019065': 'n018',
    'UnsortedMonthlyBands-Band_6-Day-2020017': 'n053',
    'UnsortedMonthlyBands-Band_7-Day-2019097': 'n056',
    'UnsortedMonthlyBands-Band_7-Day-2019321': 'n060',
    'UnsortedMonthlyBands-Band_7-Day-2019353': 'n061',
    'UnsortedMonthlyBands-Band_7-Day-2020017': 'n062',
    'UnsortedMonthlyBands-NDVI-Day-2019065': 'n063',
    'UnsortedMonthlyBands-NDVI-Day-2019097': 'n064',
    'UnsortedMonthlyBands-NDVI-Day-2019129': 'n065',
    'UnsortedMonthlyBands-NDVI-Day-2019193': 'n066',
    'UnsortedMonthlyBands-NDVI-Day-2019225': 'n067',
    'UnsortedMonthlyBands-NDVI-Day-2019321': 'n069',
    'UnsortedMonthlyBands-NDVI-Day-2019353': 'n070',
    'UnsortedMonthlyBands-NDVI-Day-2020017': 'n071',
    'Lowest6MeanBandRefl-Band_6': 'n086',
    'BandReflMedian-Band_2': 'n109',
    'BandReflMedian-Band_4': 'n111',
    'BandReflMedian-Band_2_2': 'n140',  # Changed key to make it unique
    'BandReflMin-NDVI': 'n189',
    'TempMeanGreenest3': 'n223'
}

# Create a new DataFrame with the columns we want, repeating as necessary
new_df = pd.DataFrame()
for old_col, new_col in column_mapping.items():
    if old_col in df.columns:
        new_df[new_col] = df[old_col]
    elif old_col == 'BandReflMedian-Band_2_2':  # Special handling for the duplicate
        if 'BandReflMedian-Band_2' in df.columns:
            new_df[new_col] = df['BandReflMedian-Band_2']
        else:
            print(f"Warning: Column 'BandReflMedian-Band_2' not found in the original DataFrame.")
    else:
        print(f"Warning: Column '{old_col}' not found in the original DataFrame.")

df = new_df
df = df.astype(float)
df['Y'] = 0

print(df.columns)
print(df.shape)

# Function to write ARFF chunk
def write_arff_chunk(chunk, file_number):
    arff_data = {
        'description': f'Generated from Pandas DataFrame - Chunk {file_number}',
        'relation': 'example',
        'attributes': [(col, 'NUMERIC') for col in chunk.columns],
        'data': chunk.values.tolist()
    }
    
    output_file = f"/explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics_Y_chunk_{file_number}.arff"
    with open(output_file, "w") as f:
        arff.dump(arff_data, f)
    print(f"Chunk {file_number} written to {output_file}")

# Split and write ARFF files
chunk_size = 1000000  # Adjust this value based on your memory constraints
num_chunks = math.ceil(len(df) / chunk_size)

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(df))
    chunk = df.iloc[start_idx:end_idx]
    write_arff_chunk(chunk, i+1)

print(f"Split complete. Total chunks: {num_chunks}")