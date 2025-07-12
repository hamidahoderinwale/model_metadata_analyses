import os
import sys
import json
import signal
import requests
import backoff
import threading
import time
from tqdm import tqdm
from typing import List, Dict, Any, Generator
from datasets import load_dataset, Dataset, Features, Value
from huggingface_hub import HfApi, ModelCard
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize Hugging Face API
api = HfApi()

# Constants
CHUNK_SIZE = 500
INPUT_HF_DATASET = "Eliahu/ModelAtlasData"
OUTPUT_HF_DATASET = "midah/new_uploads"
EXCLUDED_LOG_PATH = "excluded_models_secrets.log"
MAX_WORKERS_THREAD = 100 # Number of concurrent threads for API calls (may be adjusted)

# Rate limiting parameters - optimized for per-thread tracking
REQUESTS_PER_SECOND = 2  # Adjust based on your HF plan and observed limits
thread_local = threading.local()

def get_thread_rate_limiter():
    """Get or create thread-local rate limiter"""
    if not hasattr(thread_local, 'last_request_time'):
        thread_local.last_request_time = 0.0
    return thread_local.last_request_time

def rate_limited():
    """Optimized rate limiting with per-thread tracking"""
    last_time = get_thread_rate_limiter()
    now = time.time()
    elapsed = now - last_time
    min_interval = 1.0 / REQUESTS_PER_SECOND
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    thread_local.last_request_time = time.time()

# Graceful shutdown flag
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Regex-based secret detection - precompiled for speed
SECRET_PATTERNS = [
    re.compile(r"hf_[a-zA-Z0-9]{30,}"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"(?i)api[_-]?key.{0,10}[=:].{10,}"),
]

def contains_secret(text: str) -> bool:
    return any(pat.search(text) for pat in SECRET_PATTERNS)

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=10,
    factor=0.7,
    jitter=backoff.full_jitter
)
def get_model_card(model_id: str) -> str:
    rate_limited()
    try:
        # Preferred method: load the model card using ModelCard
        card = ModelCard.load(model_id)
        return card.content  # Full markdown, including YAML metadata
    except Exception:
        try:
            # Fallback: attempt to get cardData from model_info (may not always be present)
            rate_limited()
            info = api.model_info(model_id)
            card_data = getattr(info, "cardData", None)
            if card_data and isinstance(card_data, dict):
                return card_data.get("content", "N/A")
            return "N/A"
        except Exception:
            return "N/A"

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=10,
    factor=0.7,
    jitter=backoff.full_jitter
)
def get_model_metadata(model_id: str) -> Dict[str, Any] | str:
    rate_limited()
    try:
        return api.model_info(model_id).__dict__
    except Exception:
        return "N/A"

def get_output_dataset_schema() -> Features:
    return Features({
        "model_id": Value("string"),
        "metadata": Value("string"),
        "card": Value("string")
    })

def process_single_model(model_id: str) -> Dict[str, str] | None:
    """
    Fetches metadata and card for a single model_id and checks for secrets.
    Returns processed data or None if skipped/error.
    """
    if shutdown_requested:
        return None
    
    try:
        metadata = get_model_metadata(model_id)
        card = get_model_card(model_id)

        if card != "N/A" and contains_secret(card):
            print(f"Skipping {model_id}: model card appears to contain secrets.")
            with open(EXCLUDED_LOG_PATH, "a") as f:
                f.write(f"{model_id}\n")
            return None

        return {
            "model_id": model_id,
            "metadata": json.dumps(metadata, default=str) if metadata != "N/A" else "N/A",
            "card": card if card != "N/A" else "N/A"
        }
    except Exception as e:
        print(f"Error processing {model_id}: {e}")
        return None

def process_chunk_parallel(chunk_model_ids: List[str]) -> List[Dict[str, str]]:
    """
    Processes a list of model IDs in parallel using a ThreadPoolExecutor.
    Optimized with better error handling.
    """
    chunk_data = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_THREAD) as executor:
        future_to_model_id = {executor.submit(process_single_model, model_id): model_id for model_id in chunk_model_ids}
        
        for future in tqdm(as_completed(future_to_model_id), total=len(chunk_model_ids), desc="Processing models in chunk", leave=False):
            if shutdown_requested:
                break
            try:
                result = future.result()
                if result is not None:
                    chunk_data.append(result)
            except Exception as e:
                model_id = future_to_model_id[future]
                print(f"Error processing {model_id}: {e}")
    
    return chunk_data

def load_model_ids_generator() -> Generator[str, None, None]:
    """
    Generator to load model IDs one at a time to reduce memory usage.
    """
    try:
        input_dataset_stream = load_dataset(INPUT_HF_DATASET, split="train", streaming=True)
        for item in input_dataset_stream:
            if 'id' in item:
                yield item['id']
    except Exception as e:
        print(f"Error loading input dataset: {e}")
        sys.exit(1)

def main():
    print(f"Starting model extraction and upload to: {OUTPUT_HF_DATASET}")

    # --- Load model IDs efficiently ---
    print(f"Loading model IDs from {INPUT_HF_DATASET}...")
    all_model_ids = list(tqdm(load_model_ids_generator(), desc="Collecting model IDs"))
    print(f"Collected {len(all_model_ids)} model IDs from {INPUT_HF_DATASET}")

    # --- Check already uploaded splits from the hub ---
    try:
        repo_info = api.dataset_info(OUTPUT_HF_DATASET)
        uploaded_splits = [s.rfilename for s in repo_info.siblings]
        print(f"Found existing splits in {OUTPUT_HF_DATASET}. Skipping any duplicates.")
    except Exception as e:
        print(f"No existing output found or failed to list splits: {e}. Attempting to create repository.")
        try:
            api.create_repo(repo_id=OUTPUT_HF_DATASET, repo_type="dataset", exist_ok=True)
            print(f"Created/ensured output dataset: {OUTPUT_HF_DATASET}")
            uploaded_splits = []
        except Exception as create_err:
            print(f"Failed to create output dataset: {create_err}")
            sys.exit(1)

    # --- Process in chunks ---
    for i in tqdm(range(0, len(all_model_ids), CHUNK_SIZE), desc="Overall chunk progress"):
        if shutdown_requested:
            print("Graceful shutdown triggered.")
            break

        chunk_ids = all_model_ids[i:i + CHUNK_SIZE]
        split_name = f"chunk_{i}_{i + len(chunk_ids) - 1}"

        # Check if split already exists on the hub
        if any(f"/{split_name}" in s for s in uploaded_splits):
            print(f"⏭️ Split {split_name} already exists. Skipping.")
            continue

        print(f"\nProcessing chunk {i} to {i + len(chunk_ids) - 1}")
        chunk_data = process_chunk_parallel(chunk_ids)

        if not chunk_data:
            print("No data processed in this chunk or all models skipped.")
            # Add debug info
            print(f"Debug: Chunk {split_name} had {len(chunk_ids)} models but processed 0")
            continue

        # Add debug info for successful chunks
        print(f"Debug: Chunk {split_name} processed {len(chunk_data)} models out of {len(chunk_ids)}")

        # Robust upload with retries and error handling
        max_upload_retries = 5
        for attempt in range(max_upload_retries):
            try:
                chunk_ds = Dataset.from_list(chunk_data, features=get_output_dataset_schema())
                print(f"Debug: Attempting to upload {len(chunk_ds)} rows to {split_name}")
                chunk_ds.push_to_hub(
                    repo_id=OUTPUT_HF_DATASET,
                    config_name="default",
                    split=split_name,
                    commit_message=f"Add chunk {i}-{i + len(chunk_ids) - 1}"
                )
                print(f"Uploaded chunk as split: {split_name}")
                uploaded_splits.append(f"{OUTPUT_HF_DATASET}/resolve/main/data/{split_name}-00000-of-00001.parquet")
                break  # Success, exit retry loop
            except Exception as upload_err:
                print(f"Error uploading chunk {split_name} (attempt {attempt + 1}/{max_upload_retries}): {upload_err}")
                # If it's a rate limit error, wait longer before retrying
                if hasattr(upload_err, "response") and getattr(upload_err.response, "status_code", None) == 429:
                    wait_time = 60 * (attempt + 1)
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    # For other errors, use exponential backoff
                    wait_time = 5 * (2 ** attempt)
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
        else:
            print(f"Failed to upload chunk {split_name} after {max_upload_retries} attempts. Skipping this chunk.")
            with open("failed_chunks.log", "a") as f:
                f.write(f"{split_name}\n")
            continue  # Skip to next chunk

    print("All remaining chunks processed or skipped if already uploaded.")
    print(f"You can now merge splits using `concatenate_datasets` if needed.")

if __name__ == "__main__":
    main()
