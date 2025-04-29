# Script to check for missing models from folder with scrapped model trees as csvs

import os
import csv
import pandas as pd
from pathlib import Path

def normalize_name(name):
    return name.lower().strip().replace('-', '').replace('/', '').replace('_', '')

def extract_model_base_name(filename):
    filename = filename.strip('"')
    parts = filename.split('_finetunes_')
    if parts and parts[0]:
        return parts[0]
    if filename.endswith('.csv'):
        return filename[:-4]
    return filename

def extract_org_and_model(model_id):
    parts = model_id.split('/')
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, model_id.strip()

def find_missing_models(model_filename_list, model_id_list):
    model_names = []
    for filename in model_filename_list:
        if filename.strip():
            base_name = extract_model_base_name(filename)
            model_names.append(base_name)

    normalized_model_names = set(normalize_name(name) for name in model_names)
    missing_models = []
    found_models = []

    for model_id in model_id_list:
        if not model_id.strip():
            continue
        org, model_name = extract_org_and_model(model_id)
        normalized_model = normalize_name(model_name)

        if normalized_model in normalized_model_names:
            found_models.append(model_id)
            continue

        match_found = False
        for name in normalized_model_names:
            if normalized_model in name:
                found_models.append(model_id)
                match_found = True
                break

        if not match_found:
            missing_models.append(model_id)

    return missing_models, found_models, model_names

def generate_files_csv(source_folder, output_csv_path):
    file_names = [f for f in os.listdir(source_folder) if f.endswith('.csv')]
    df = pd.DataFrame({'id': file_names})
    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(file_names)} filenames to {output_csv_path}")
    return file_names  # So we don’t read it again later

def read_model_ids(filepath):
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row.get('id')
            if model_id and model_id.strip():
                entries.append(model_id.strip())
    return entries

def main():
    # Paths
    source_folder = '/home/topos/Desktop/HFScrapingProj/v2_datasets/csv' # folder with model tree csvs to check
    filenames_csv = f'{source_folder}/files.csv' # provide file name to save as csv file list 
    model_ids_csv = '_model_list.csv' # provide list of models to compare against (e.g. produced with `get_top_models.py`)
    output_full_csv = 'path_to/missing_models_output.csv' # provide name of desired output csv for full model list (each model is checked as missing/found)
    output_missing_csv = 'path_to/only_missing_models.csv' # provide name of desired output csv for only the missing models

    # Step 1: Generate files.csv
    model_filename_list = generate_files_csv(source_folder, filenames_csv)

    # Step 2: Read model IDs
    model_id_list = read_model_ids(model_ids_csv)

    # Step 3: Compare and find missing
    missing_models, found_models, extracted_names = find_missing_models(model_filename_list, model_id_list)

    # Print summary
    print(f"Total filenames processed: {len(model_filename_list)}")
    print(f"Total model IDs processed: {len(model_id_list)}")
    print(f"Models found: {len(found_models)}")
    print(f"Models missing: {len(missing_models)}")
    if model_id_list:
        print(f"Percentage found: {100 * len(found_models) / len(model_id_list):.2f}%")

    print("\nMissing models:")
    for model in missing_models:
        print(f"  - {model}")

    # Save full results
    with open(output_full_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model ID', 'Status'])
        for model_id in model_id_list:
            status = "Found" if model_id in found_models else "Missing"
            writer.writerow([model_id, status])
    print(f"\nFull results saved to {output_full_csv}")

    # Save only missing
    with open(output_missing_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Missing Model ID'])
        for model in missing_models:
            writer.writerow([model])
    print(f"Only missing models saved to {output_missing_csv}")

if __name__ == "__main__":
    main()
