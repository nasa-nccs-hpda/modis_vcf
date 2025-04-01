import sys
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import geopandas as gpd
import arff


import pandas as pd
import numpy as np
import arff



# Load Parquet file
# df = pd.read_parquet('/explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics_subset.parquet')
# df = pd.read_parquet('/explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics.parquet')
df = pd.read_parquet(sys.argv[1])
df = df.drop(['x', 'y', 'x_idx', 'y_idx'], axis=1, errors='ignore')

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

# Check if we have all the columns we expect
expected_columns = set(column_mapping.values())
actual_columns = set(new_df.columns)
missing_columns = expected_columns - actual_columns

if missing_columns:
    print(f"Warning: The following columns are missing: {missing_columns}")

# Now new_df has all the columns we want, with the new names

# If you want to keep only these columns in the original DataFrame:
df = new_df
df = df.astype(float)

if 'Y' not in df.columns:
    df['Y'] = 0

print(df.columns)
print(df.shape)


for col in df.columns:
    if isinstance(df[col], pd.DataFrame):
        print(f"Column '{col}' is a DataFrame")
    else:
        print(f"Column '{col}' is a {type(df[col]), df[col].dtype}")



# Convert DataFrame to ARFF dictionary format
arff_data = {
    'description': 'Generated from Pandas DataFrame',
    'relation': 'example',
    'attributes': [(col, 'NUMERIC' if df[col].dtype in [np.float64, np.int64] else list(df[col].unique())) for col in df.columns],
    'data': df.values.tolist()
}

# Save to ARFF file
# "/explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics_Y.arff"
output_file = sys.argv[2]
with open(output_file, "w") as f:
    arff.dump(arff_data, f)

