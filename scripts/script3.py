!pip install adapters
!pip install backoff # handle rate-limiting
import requests
import pandas as pd
import datetime
import json
import csv
from huggingface_hub import HfApi
from bs4 import BeautifulSoup
from adapters import list_adapters
from huggingface_hub import hf_hub_download
from adapters import AutoAdapterModel
import re
import backoff

# Initialize API
api = HfApi()

# Function to validate Hugging Face model URL
def validate_hf_model_url(url):
    pattern = r"^https://huggingface.co/([\w\-]+)/([\w\-]+)$"
    match = re.match(pattern, url)
    return match.groups() if match else None

# Apply backoff to requests
@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=5,
    factor=2,  # Base of exponential backoff
    jitter=backoff.full_jitter  # Add randomness to prevent thundering herd
)
def make_http_request(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise exception for non-200 status codes
    return response

# Page with model finetunes
@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=5,
    factor=2,
    jitter=backoff.full_jitter
)
def get_finetuned_models_page(model_org, model_name):
    all_model_links = []  # Store all links across pages
    page_num = 0
    while True:
        try:
            search_url = f"https://huggingface.co/models?other=base_model:finetune:{model_org}/{model_name}&p={page_num}"
            response = make_http_request(search_url)

            soup = BeautifulSoup(response.text, "html.parser")
            model_divs = soup.find_all("div", class_="w-full truncate")
            if not model_divs:
                break  # Exit if no more models on the page

            for div in model_divs:
                header = div.find("header")
                if header:
                    model_link = header.get("title")
                    if model_link:
                        all_model_links.append(f"https://huggingface.co/{model_link}")

            page_num += 1  # Move to the next page
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            break

    return all_model_links

# Get model card data
@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=5,
    factor=2,
    jitter=backoff.full_jitter
)
def get_model_card(model_id):
    try:
        # Try to download model card if available
        readme_path = hf_hub_download(repo_id=model_id, filename='README.md')
        with open(readme_path, 'r', encoding='utf-8') as f:
            card_content = f.read()
        return card_content
    except Exception as e:
        print(f"  Could not download model card: {str(e)[:100]}...")
        return ""

# Get model metadata with backoff
@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=5,
    factor=2,
    jitter=backoff.full_jitter
)
def get_model_metadata(model_id):
    return api.model_info(model_id).__dict__

# Truncate metadata
def filter_metadata(json_metadata):
    keys_to_keep = ["modelId", "sha", "tags", "downloads", "pipeline_tag"]
    return {k: json_metadata.get(k) for k in keys_to_keep if k in json_metadata}

# Get adapter models
@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=5,
    factor=2,
    jitter=backoff.full_jitter
)
def get_adapter_models_page(model_org, model_name):
    all_adapter_links = []  # Store all adapter links across pages
    page_num = 0
    while True:
        try:
            search_url = f"https://huggingface.co/models?other=base_model:adapter:{model_org}/{model_name}&p={page_num}"
            response = make_http_request(search_url)

            soup = BeautifulSoup(response.text, "html.parser")
            model_divs = soup.find_all("div", class_="w-full truncate")
            if not model_divs:
                break  # Exit if no more models on the page

            for div in model_divs:
                header = div.find("header")
                if header:
                    model_link = header.get("title")
                    if model_link:
                        all_adapter_links.append(f"https://huggingface.co/{model_link}")

            page_num += 1  # Move to the next page
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            break

    return all_adapter_links

# Recursive DFS (depth-first search) for finding fine-tunes
def dfs_finetunes(model_url, visited, depth=0, results=None):
    if results is None:
        results = []

    if model_url in visited:
        return results
    visited.add(model_url)

    validated = validate_hf_model_url(model_url)
    if not validated:
        print(f"Invalid URL skipped: {model_url}")
        return results

    model_org, model_name = validated
    model_id = f"{model_org}/{model_name}"

    print(f"\n{'  ' * depth}Fetching metadata for: {model_id}")
    try:
        model_metadata = get_model_metadata(model_id)
        filtered_metadata = filter_metadata(model_metadata)
        json_metadata = json.dumps(filtered_metadata, default=str)
        model_card = get_model_card(model_id)

        finetune_links = get_finetuned_models_page(model_org, model_name)
        # Removing Duplicate Children
        finetune_links = list(set(finetune_links))
        print(f"{'  ' * depth}Found {len(finetune_links)} fine-tunes at depth {depth}.")

        adapter_links = get_adapter_models_page(model_org, model_name)
        print(f"{'  ' * depth}Found {len(adapter_links)} adapter models for {model_id}.")

        results.append({
            "model_id": model_id,
            "card": model_card,
            "metadata": json_metadata,
            "depth": depth,
            "children": finetune_links,
            "children_count": len(finetune_links),
            "adapters": adapter_links,
            "adapters_count": len(adapter_links)
        })

        for link in finetune_links:
            dfs_finetunes(link, visited, depth + 1, results)
    except Exception as e:
        print(f"Error processing {model_id}: {e}")

    return results

# Timestamp for the run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Function to save results as JSON
def save_json(results, model_name):
    filename = f"{model_name}_finetunes_{timestamp}.json"
    data = {
        "models": results
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, default=str)
    print(f"Results saved to {filename}")

# Function to save results as CSV (pandas)
def save_csv(results, model_name):
    filename = f"{model_name}_finetunes_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(filename, index=True)
    print(f"Results saved to {filename}")

# Main execution
if __name__ == "__main__":
    model_url = input("Enter the Hugging Face model URL: ").strip()
    visited = set()
    results = dfs_finetunes(model_url, visited)

    if results:
        model_name = results[0]["model_id"].split("/")[-1]  # Extract model name for file naming
        save_json(results, model_name)
        save_csv(results, model_name)
    else:
        print("No fine-tuned models found.")

'''Links for testing: https://huggingface.co/NousResearch/DeepHermes-3-Mistral-24B-Preview (3 fine-tunes at depth 0, 1 fine-tune at depth 1 for 'AlSamCur123/DeepHermes-3-Mistral-24BContinuedFine')
https://huggingface.co/perplexity-ai/r1-1776 (11 fine-tunes at depth 0)'''
