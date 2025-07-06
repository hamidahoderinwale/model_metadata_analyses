import json
import re
from collections import Counter, defaultdict

# === Step 1: Extract file list from model metadata siblings ===
def extract_repo_files_from_metadata(metadata_str):
    """
    Extracts a list of file names from the 'siblings' field within the model's metadata.
    The 'siblings' field is expected to be a string representation of a list of RepoSibling objects.

    Args:
        metadata_str (str): The raw string content of the 'metadata' field from a dataset entry.

    Returns:
        list: A deduplicated list of file names found in the 'siblings' list, or an empty list
              if parsing fails or no siblings are found.
    """
    if not isinstance(metadata_str, str):
        return []

    try:
        # The 'metadata' field is a string representation of a JSON object.
        # We need to parse it first.
        metadata = json.loads(metadata_str)
        
        # The 'siblings' field contains strings like "RepoSibling(rfilename='filename.ext', ...)"
        siblings_list_str = metadata.get("siblings", [])

        file_list = []
        for sibling_str in siblings_list_str:
            # Use regex to extract the filename from 'rfilename='
            match = re.search(r"rfilename='([^']+)'", sibling_str)
            if match:
                file_list.append(match.group(1))
        return list(set(file_list))  # deduplicate
    except json.JSONDecodeError:
        print(f"Warning: Could not decode metadata JSON. Skipping entry. Metadata snippet: {metadata_str[:100]}...")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during file extraction: {e}. Metadata snippet: {metadata_str[:100]}...")
        return []

# === Step 2: Check if required files are mentioned ===
def can_analyze_architecture(file_list):
    """
    Checks if 'config.json' and 'model.safetensors.index.json' are present in the provided file list.

    Args:
        file_list (list): A list of file names extracted from a model's metadata.

    Returns:
        bool: True if both 'config.json' and 'model.safetensors.index.json' are found, False otherwise.
    """
    return "config.json" in file_list and "model.safetensors.index.json" in file_list

# === Step 3: Simulate the architecture analysis ===
def analyze_architecture_simulation(model_id, file_list):
    """
    Simulates architecture analysis by confirming required files are listed.
    In a real scenario, you would typically download these files to perform actual analysis.

    Args:
        model_id (str): The unique identifier of the model.
        file_list (list): The list of files associated with the model.

    Returns:
        bool: True if architecture analysis is considered "possible" (i.e., required files are listed).
    """
    if can_analyze_architecture(file_list):
        print(f"  - Model '{model_id}': 'config.json' and 'model.safetensors.index.json' are listed. Architecture analysis possible (simulated).")
        return True
    else:
        print(f"  - Model '{model_id}': Required files for architecture analysis are missing.")
        return False

# === Main processing logic for the dataset ===
def process_huggingface_dataset(dataset_rows):
    """
    Processes each row of a Hugging Face dataset to extract file information
    and check for the presence of specific files.

    Args:
        dataset_rows (list of dict or list of list): The dataset content.
            Each item in the list should represent a model entry.
            If it's a list of lists (e.g., from CSV parsing),
            assume the order is 'model_id', 'gated', 'card', 'metadata', ...
            If it's a list of dictionaries (e.g., from `datasets` library),
            assume keys like 'model_id' and 'metadata' exist.

    Returns:
        defaultdict: A dictionary where keys are model_ids and values are
                     dictionaries containing 'files_listed' and 'can_analyze_architecture' status.
    """
    results = defaultdict(dict)

    for row in dataset_rows:
        model_id = None
        metadata_str = None

        if isinstance(row, dict):
            # Assuming row is a dictionary from datasets library
            model_id = row.get("model_id")
            metadata_str = row.get("metadata")
        elif isinstance(row, list):
            # Assuming row is a list from CSV parsing
            if len(row) > 3: # Ensure 'metadata' column exists (index 3)
                model_id = row[0]
                metadata_str = row[3] # 'metadata' is the 4th column (index 3)
        else:
            print(f"Warning: Unexpected row format. Skipping row: {row}")
            continue

        if not model_id or not metadata_str:
            print(f"Warning: Missing 'model_id' or 'metadata' in row. Skipping entry: {row}")
            continue

        print(f"\nProcessing model: {model_id}")
        
        file_list = extract_repo_files_from_metadata(metadata_str)
        print(f"  - Extracted files: {file_list}")

        analysis_possible = analyze_architecture_simulation(model_id, file_list)
        results[model_id]["files_listed"] = file_list
        results[model_id]["can_analyze_architecture"] = analysis_possible

    return results

# Example of how you would use this with a real dataset:
if __name__ == "__main__":
    # --- METHOD 1: Loading from a CSV file ---
    # Assuming you have your dataset in a CSV file named 'huggingface_models.csv'
    # with columns like 'model_id', 'gated', 'card', 'metadata', ...
    #
    # import csv
    #
    # dataset_rows_from_csv = []
    # try:
    #     with open('huggingface_models.csv', 'r', encoding='utf-8') as f:
    #         reader = csv.reader(f)
    #         header = next(reader) # Skip header row
    #         for row in reader:
    #             dataset_rows_from_csv.append(row)
    #     print("Processing data loaded from CSV...")
    #     processed_results_csv = process_huggingface_dataset(dataset_rows_from_csv)
    #     print("\n--- CSV Processing Summary ---")
    #     for model_id, data in processed_results_csv.items():
    #         print(f"Model ID: {model_id}")
    #         print(f"  Files listed: {data['files_listed']}")
    #         print(f"  Can analyze architecture: {data['can_analyze_architecture']}")
    # except FileNotFoundError:
    #     print("Error: 'huggingface_models.csv' not found. Please create the file or adjust the path.")
    # except Exception as e:
    #     print(f"An error occurred while reading the CSV: {e}")


    # --- METHOD 2: Using Hugging Face's datasets library (recommended for HF datasets) ---
    # If your data is in a Hugging Face dataset (e.g., from the Hub)
    # You would typically install it: pip install datasets
    #
    # from datasets import load_dataset
    #
    # try:
    #     # Replace 'your_dataset_name' and 'your_config_name' with actual values
    #     # For example, if you downloaded the model metadata directly as a JSONL or similar,
    #     # you might load it as a local file, or if it's a specific dataset on the Hub.
    #     # This is a placeholder for how you would load your specific dataset.
    #     # Example for a hypothetical dataset:
    #     # dataset = load_dataset('some_org/some_model_metadata_dataset', split='train')
    #
    #     # If your dataset is a local JSONL file, for instance:
    #     # dataset = load_dataset('json', data_files='your_local_model_metadata.jsonl', split='train')
    #
    #     # Placeholder for demonstration if you don't have a specific HF dataset ready:
    #     print("\n--- To use the 'datasets' library, uncomment and configure the code below. ---")
    #     print("Example: dataset = load_dataset('some_org/some_model_metadata_dataset', split='train')")
    #     print("Or for a local JSONL: dataset = load_dataset('json', data_files='your_local_model_metadata.jsonl', split='train')")
    #
    #     # Example of how you'd iterate if 'dataset' was loaded:
    #     # dataset_rows_from_hf = []
    #     # for entry in dataset:
    #     #     dataset_rows_from_hf.append(entry) # Assuming each entry is a dict
    #
    #     # print("Processing data loaded from Hugging Face datasets library...")
    #     # processed_results_hf = process_huggingface_dataset(dataset_rows_from_hf)
    #     # print("\n--- Hugging Face Datasets Summary ---")
    #     # for model_id, data in processed_results_hf.items():
    #     #     print(f"Model ID: {model_id}")
    #     #     print(f"  Files listed: {data['files_listed']}")
    #     #     print(f"  Can analyze architecture: {data['can_analyze_architecture']}")
    #
    # except ImportError:
    #     print("\n'datasets' library not found. Install with: pip install datasets")
    # except Exception as e:
    #     print(f"An error occurred while loading/processing the Hugging Face dataset: {e}")

    print("\nScript execution finished. Provide your dataset rows (e.g., from CSV or Hugging Face datasets) to the 'process_huggingface_dataset' function.")
