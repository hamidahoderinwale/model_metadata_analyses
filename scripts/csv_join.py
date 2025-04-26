# Script to join CSVs.

import pandas as pd
import glob

# Set the path to your main datasets folder
path = "/v2_datasets"

# Find all CSV files recursively in all subfolders
all_files = glob.glob(path + '/**/*.csv', recursive=True)
all_files.sort()

combined_df = pd.DataFrame()

for i, fname in enumerate(all_files):
    if i == 0:
        # Read the first file with headers
        temp_df = pd.read_csv(fname)
    else:
        # Read subsequent files, skipping their headers
        temp_df = pd.read_csv(fname, header=0)
    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

# Drop rows that are completely empty
combined_df = combined_df.dropna(how='all')

# Save to output file
combined_df.to_csv('joined.csv', index=False)
