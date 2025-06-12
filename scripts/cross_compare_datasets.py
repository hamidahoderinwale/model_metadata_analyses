import pandas as pd
import os
import json
import logging
from datasets import load_dataset
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cross_compare_datasets(hf_dataset_name_1: str, hf_dataset_file_1: str, hf_dataset_name_2: str, hf_dataset_file_2: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_output_path = os.path.join(output_dir, f"dataset_comparison_{timestamp}")
    os.makedirs(analysis_output_path, exist_ok=True)

    logger.info(f"Loading Hugging Face dataset: {hf_dataset_name_1}/{hf_dataset_file_1}")
    try:
        hf_dataset_1 = load_dataset(hf_dataset_name_1, hf_dataset_file_1)
        hf_df_1 = hf_dataset_1['train'].to_pandas()
        hf_model_ids_1 = set(hf_df_1['model_id'].dropna().tolist())
        logger.info(f"Loaded {len(hf_model_ids_1)} unique model IDs from the first Hugging Face dataset.")
    except Exception as e:
        logger.error(f"Error loading first Hugging Face dataset: {e}")
        return

    logger.info(f"Loading Hugging Face dataset: {hf_dataset_name_2}/{hf_dataset_file_2}")
    try:
        hf_dataset_2 = load_dataset(hf_dataset_name_2, hf_dataset_file_2)
        hf_df_2 = hf_dataset_2['train'].to_pandas()
        hf_model_ids_2 = set(hf_df_2['id'].dropna().tolist())
        logger.info(f"Loaded {len(hf_model_ids_2)} unique model IDs from the second Hugging Face dataset.")
    except Exception as e:
        logger.error(f"Error loading second Hugging Face dataset: {e}")
        return

    logger.info("Performing cross-comparison...")
    
    common_model_ids = sorted(list(hf_model_ids_1.intersection(hf_model_ids_2)))
    hf1_only_model_ids = sorted(list(hf_model_ids_1.difference(hf_model_ids_2)))
    hf2_only_model_ids = sorted(list(hf_model_ids_2.difference(hf_model_ids_1)))

    # Convert lists to DataFrames for saving
    common_df = pd.DataFrame(common_model_ids, columns=['model_id'])
    hf1_only_df = pd.DataFrame(hf1_only_model_ids, columns=['model_id'])
    hf2_only_df = pd.DataFrame(hf2_only_model_ids, columns=['model_id'])

    # Save results to CSVs
    common_csv_path = os.path.join(analysis_output_path, 'common_model_ids.csv')
    hf1_only_csv_path = os.path.join(analysis_output_path, 'hf1_only_model_ids.csv')
    hf2_only_csv_path = os.path.join(analysis_output_path, 'hf2_only_model_ids.csv')

    common_df.to_csv(common_csv_path, index=False)
    hf1_only_df.to_csv(hf1_only_csv_path, index=False)
    hf2_only_df.to_csv(hf2_only_csv_path, index=False)

    logger.info(f"Comparison complete. Results saved to {analysis_output_path}/")
    logger.info(f"- Common models: {len(common_model_ids)}")
    logger.info(f"- Models only in first Hugging Face dataset: {len(hf1_only_model_ids)}")
    logger.info(f"- Models only in second Hugging Face dataset: {len(hf2_only_model_ids)}")

    # Optionally, save a summary JSON
    summary_stats = {
        "total_hf1_models": len(hf_model_ids_1),
        "total_hf2_models": len(hf_model_ids_2),
        "common_models_count": len(common_model_ids),
        "hf1_only_models_count": len(hf1_only_model_ids),
        "hf2_only_models_count": len(hf2_only_model_ids),
        "common_models_file": common_csv_path,
        "hf1_only_models_file": hf1_only_csv_path,
        "hf2_only_models_file": hf2_only_csv_path,
    }

    summary_json_path = os.path.join(analysis_output_path, 'comparison_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(summary_stats, f, indent=4)
    logger.info(f"Summary statistics saved to {summary_json_path}")

if __name__ == "__main__":
    # Define your Hugging Face dataset details
    HF_DATASET_NAME_1 = "midah/removed_gemma_trees"
    HF_DATASET_FILE_1 = "default"  # Adjust as necessary
    HF_DATASET_NAME_2 = "Eliahu/ModelAtlasData"  # Updated to the new dataset
    HF_DATASET_FILE_2 = "default"  # Adjust as necessary for the second dataset

    # Define the output directory relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.join(script_dir, "outputs")
    OUTPUT_DIR = os.path.join(output_root, "dataset_cross_comparison")

    cross_compare_datasets(HF_DATASET_NAME_1, HF_DATASET_FILE_1, HF_DATASET_NAME_2, HF_DATASET_FILE_2, OUTPUT_DIR) 