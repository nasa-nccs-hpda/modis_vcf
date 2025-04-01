import arff

# Read the original ARFF file
with open('/explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics.arff', 'r') as f:
    dataset = arff.load(f)

# Add a dummy class attribute
dataset['attributes'].append(('Y', 'NUMERIC'))

# Add a dummy value (e.g., 0) for each instance
for row in dataset['data']:
    row.append(0)

# Write the modified ARFF file
with open('/explore/nobackup/projects/ilab/scratch/jacaraba/notebooks/vcf_rf_test/MOD44C_h20v06_2019_3-Metrics_Class.arff', 'w') as f:
    arff.dump(dataset, f)

print("Modified ARFF file created: modified_dataset.arff")
