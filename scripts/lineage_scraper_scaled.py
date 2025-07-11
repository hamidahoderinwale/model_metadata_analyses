# [draft] model relation append script with asyncio and other optimizations for scale 

import asyncio
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm as tqdm_asyncio
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
import os

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("midah/removed_gemma_trees")

INPUT_DATASET_ID = ds
OUTPUT_DATASET_ID = "midah/lineage_output_async"
LIMIT = None  # Set to an int for testing, or None to process all models
MAX_CONCURRENT_REQUESTS = 50
TEMP_UPLOAD_DIR = "upload_tmp"

REL_TYPE_MAP = {
    "finetuned": "finetuned",
    "finetunes": "finetuned",
    "quantized": "quantized",
    "adapter": "adapter",
    "adapters": "adapter",
    "merge": "merged",
    "merges": "merged"
}

# parse hf pages for lineage
def extract_lineage_rows_from_html(html, model_id):
    soup = BeautifulSoup(html, "html.parser")
    base_model_id = None
    lineage_rows = []

    base_model_section = soup.find("p", string=lambda s: s and s.strip().lower() == "base model")
    if not base_model_section:
        return []

    tree_container = base_model_section.find_parent("div", class_="text-smd")
    if not tree_container:
        return []

    flex_div = base_model_section.find_parent("div", class_="flex")
    if flex_div:
        base_model_link = flex_div.find("a", href=True)
        if base_model_link:
            base_model_id = base_model_link.text.strip()

    for flex_div in tree_container.find_all("div", class_="flex", recursive=False):
        label_div = flex_div.find("div", class_="font-semibold")
        if not label_div:
            continue

        rel_type_raw = label_div.get_text(separator=" ", strip=True).lower()
        rel_type = next((REL_TYPE_MAP[key] for key in REL_TYPE_MAP if key in rel_type_raw), None)
        if not rel_type:
            continue

        for a in flex_div.find_all("a", href=True):
            href = a["href"]
            if "base_model:" in href:
                try:
                    parent_id = href.split("base_model:")[-1].split(":")[-1]
                except IndexError:
                    continue
            else:
                parent_id = a.text.strip()

            if parent_id == model_id or "/" not in parent_id:
                continue

            lineage_rows.append({
                "model_id": model_id,
                "parent_model_id": parent_id,
                "parent_relationship": rel_type,
                "base_model_id": base_model_id
            })

    return lineage_rows

# scrape hf pages for lineage
async def fetch_and_process(session, model_id):
    try:
        url = f"https://huggingface.co/{model_id}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ModelLineageBot/1.0)"}
        async with session.get(url, timeout=20) as resp:
            if resp.status == 200:
                html = await resp.text()
                return extract_lineage_rows_from_html(html, model_id)
    except Exception as e:
        print(f"Error fetching {model_id}: {e}")
    return []

async def process_models_concurrently(model_ids, max_concurrent_requests=50):
    results = []
    connector = aiohttp.TCPConnector(limit=max_concurrent_requests)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [fetch_and_process(session, model_id) for model_id in model_ids]
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Scraping models"):
            lineage = await coro
            if lineage:
                results.extend(lineage)
    return results

# input models from dataset
def collect_model_ids(dataset_id, limit=None):
    ds_streaming = load_dataset(dataset_id, split="train", streaming=True)
    ids = []
    for i, row in enumerate(ds_streaming):
        if limit and i >= limit:
            break
        ids.append(row["model_id"])
    return ids

def load_existing_model_ids(dataset_id):
    try:
        ds = load_dataset(dataset_id, split="train")
        return set(ds["model_id"])
    except Exception:
        return set()

def upload_incremental_to_hub(df, dataset_id, tmp_dir="upload_tmp"):
    if len(df) == 0:
        print("No new rows to upload.")
        return
    os.makedirs(tmp_dir, exist_ok=True)
    new_ds = Dataset.from_pandas(df)
    new_ds.save_to_disk(tmp_dir)
    print(f"Pushing {len(new_ds)} rows to {dataset_id}...")
    new_ds.push_to_hub(dataset_id)
    print("Upload complete.")

# process models
async def main():
    print("Loading input models...")
    all_model_ids = collect_model_ids(INPUT_DATASET_ID, limit=LIMIT)
    print(f"Total models loaded: {len(all_model_ids)}")

    print("Checking existing uploads...")
    existing_model_ids = load_existing_model_ids(OUTPUT_DATASET_ID)
    model_ids = [mid for mid in all_model_ids if mid not in existing_model_ids]
    print(f"Models to process: {len(model_ids)}")

    if not model_ids:
        print("All models already processed.")
        return

    lineage_data = await process_models_concurrently(model_ids, max_concurrent_requests=MAX_CONCURRENT_REQUESTS)
    if not lineage_data:
        print("No new lineage data extracted.")
        return

    df = pd.DataFrame(lineage_data)
    df.drop_duplicates(subset=["model_id", "parent_model_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    upload_incremental_to_hub(df, OUTPUT_DATASET_ID, tmp_dir=TEMP_UPLOAD_DIR)

# === ENTRYPOINT ===
if __name__ == "__main__":
    asyncio.run(main())
