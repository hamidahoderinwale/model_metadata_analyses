# Script to join csvs into one csv with all scrapped trees

import pandas as pd
import glob
import os
from tqdm import tqdm

# Set your folder directly
path = "_path_to_folder_with_csvs"

# Find all CSV files (non-recursive, or recursive if needed)
all_files = glob.glob(os.path.join(path, '*.csv'))
all_files.sort()

if not all_files:
    print(" No CSV files found. Please check the folder path.")
    exit(1)

print(f" Found {len(all_files)} CSV files. Merging with single header...")

# Initialize the combined DataFrame
combined_df = None

for i, fname in enumerate(tqdm(all_files, desc="Merging CSVs", unit="file")):
    try:
        temp_df = pd.read_csv(fname)
        
        # Optionally track the source file (you can remove this line if not needed)
        temp_df['source_file'] = os.path.basename(fname)

        if combined_df is None:
            combined_df = temp_df
        else:
            combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    except Exception as e:
        print(f"Skipped {fname}: {e}")

# Save outputs
csv_output = '_all_model_trees.csv'
parquet_output = '_all_model_trees.parquet'

combined_df.to_csv(csv_output, index=False)
combined_df.to_parquet(parquet_output, index=False)

print(f"Done! Saved merged dataset as '{csv_output}' and '{parquet_output}'.")
