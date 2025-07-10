import os
import json
from datasets import load_dataset, Dataset, Features, Value
from huggingface_hub import HfApi
from typing import List, Dict, Any
from tqdm import tqdm

# Constants
INPUT_HF_DATASET = "Eliahu/ModelAtlasData"
OUTPUT_HF_DATASET = "midah/new_uploads"
CHUNK_SIZE = 500
EXCLUDED_LOG_PATH = "excluded_models_secrets.log"

# Initialize API
api = HfApi()

# Secret regex patterns
def contains_secret(text: str) -> bool:
    import re
    patterns = [
        r"hf_[a-zA-Z0-9]{30,}",
        r"sk-[a-zA-Z0-9]{20,}",
        r"(?i)api[_-]?key.{0,10}[=:].{10,}",
    ]
    return any(re.search(pat, text) for pat in patterns)

# Get already uploaded model IDs from all splits
def get_uploaded_model_ids(output_dataset: str) -> set:
    uploaded_ids = set()
    try:
        repo_info = api.dataset_info(output_dataset)
        splits = [s.rfilename.replace(".jsonl", "") for s in repo_info.siblings if s.rfilename.endswith(".jsonl")]
        for split in splits:
            try:
                ds = load_dataset(output_dataset, split=split)
                uploaded_ids.update(ds["model_id"])
            except Exception as e:
                print(f"⚠️ Failed loading split {split}: {e}")
    except Exception as e:
        print(f"⚠️ Could not fetch repo info for {output_dataset}: {e}")
    return uploaded_ids

# Get remaining models to process
def get_remaining_model_ids(input_dataset: str, already_uploaded_ids: set) -> List[str]:
    remaining = []
    for item in load_dataset(input_dataset, split="train", streaming=True):
        if 'id' in item and item['id'] not in already_uploaded_ids:
            remaining.append(item['id'])
    return remaining

# Dummy API metadata + card fetch (placeholder for backoff-wrapped calls)
def get_model_metadata(model_id: str) -> Dict[str, Any]:
    try:
        return api.model_info(model_id).__dict__
    except Exception:
        return {}

def get_model_card(model_id: str) -> str:
    try:
        info = api.model_info(model_id)
        return getattr(info, "cardData", {}).get("content", "N/A")
    except Exception:
        return "N/A"

# Push chunk
def push_chunk(model_ids: List[str], start_idx: int):
    chunk_data = []
    for model_id in model_ids:
        metadata = get_model_metadata(model_id)
        card = get_model_card(model_id)
        if card != "N/A" and contains_secret(card):
            with open(EXCLUDED_LOG_PATH, "a") as f:
                f.write(f"{model_id}\n")
            continue
        chunk_data.append({
            "model_id": model_id,
            "metadata": json.dumps(metadata, default=str),
            "card": card
        })
    if not chunk_data:
        return
    split_name = f"chunk_{start_idx}_{start_idx + len(model_ids) - 1}"
    features = Features({
        "model_id": Value("string"),
        "metadata": Value("string"),
        "card": Value("string")
    })
    dataset = Dataset.from_list(chunk_data, features=features)
    dataset.push_to_hub(OUTPUT_HF_DATASET, split=split_name, config_name="default")

# Main logic
def main():
    uploaded_ids = get_uploaded_model_ids(OUTPUT_HF_DATASET)
    remaining_ids = get_remaining_model_ids(INPUT_HF_DATASET, uploaded_ids)
    print(f"{len(remaining_ids)} models left to process.")

    for i in tqdm(range(0, len(remaining_ids), CHUNK_SIZE)):
        chunk = remaining_ids[i:i + CHUNK_SIZE]
        push_chunk(chunk, i)

main()
