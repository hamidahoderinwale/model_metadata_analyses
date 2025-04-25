# Multi-model runner for HuggingFace model analysis
import requests
import pandas as pd
import datetime
import json
import csv
import os
import time
import backoff
from huggingface_hub import HfApi
from bs4 import BeautifulSoup
import re
import huggingface_hub as hf

api = HfApi()

# Import from script3
try:
    from script3 import dfs_finetunes, save_json, save_csv
    print("Successfully imported functions from script3")
except ImportError as e:
    raise ImportError(f"Failed to import functions from script3: {e}. Make sure script3.py exists and contains the required functions.")

# Timestamp for output files
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Enhanced processing with backoff
@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=5,
    max_time=60,
    factor=2,
    jitter=backoff.full_jitter
)

def process_single_model(model_id, output_dir):
    """Process individual model with error handling"""

    output_dir = "v2_datasets"
    model_url = f"https://huggingface.co/{model_id}"
    visited = set()

    print(f"Starting analysis for {model_id}")
    start_time = time.time()

    results = dfs_finetunes(model_url, visited)

    if results:
        model_name = model_id.split("/")[-1]
        json_path = os.path.join(output_dir, f"{model_name}_finetunes_{timestamp}.json")
        csv_path = os.path.join(output_dir, f"{model_name}_finetunes_{timestamp}.csv")

        save_json(results, json_path)
        save_csv(results, csv_path)

        elapsed = time.time() - start_time
        print(f"Completed {model_name} in {elapsed:.2f}s")
        return True, model_name
    else:
        elapsed = time.time() - start_time
        print(f"No results for {model_id} after {elapsed:.2f}s")
        return False, model_id

def process_models_from_csv(csv_path, output_dir="results", delay=2, limit=1000):
    """Process models from CSV with validation"""
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if 'id' not in df.columns:
        raise ValueError("CSV must contain a column named 'id'.")

    # Validate model IDs
    df = df.head(limit).copy()
    df['is_valid'] = df['id'].apply(lambda x: bool(re.match(r'^[\w-]+\/[\w-]+$', str(x))))
    invalid_models = df[~df['is_valid']]

    if not invalid_models.empty:
        print(f"Warning: {len(invalid_models)} invalid model IDs detected:")
        print(invalid_models['id'].tolist())
        print("Skipping invalid models...")

    df = df[df['is_valid']]
    total_models = len(df)

    successful = 0
    failed = 0

    print(f"Starting analysis of {total_models} valid models")

    for idx, row in df.iterrows():
        model_id = row['id'].strip()
        print(f"\n[{idx + 1}/{total_models}] Processing: {model_id}")

        try:
            success, name = process_single_model(model_id, output_dir)

            if success:
                successful += 1
            else:
                failed += 1

        except Exception as e:
            print(f"Critical error with {model_id}: {str(e)[:100]}")
            failed += 1

        if idx < total_models - 1:
            print(f"Waiting {delay} seconds...")
            time.sleep(delay)

    print(f"\nAnalysis Summary:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    print("HuggingFace Multi-Model Analysis Tool")
    print("====================================")

    csv_input = input("Path to CSV with model IDs: ").strip()

    try:
        process_models_from_csv(csv_input)
    except Exception as e:
        print(f"Initialization error: {e}")

    
