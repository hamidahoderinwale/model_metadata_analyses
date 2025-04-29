# Takes a csv of a list of models (e.g. from `get_top_models.py` and return csv and json of their trees)

import requests
import pandas as pd
import datetime
import json
import re
import backoff
import os
from huggingface_hub import HfApi, hf_hub_download
from bs4 import BeautifulSoup
from collections import deque

# Initialize Hugging Face API
api = HfApi()
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ------------- Helpers -------------

def validate_hf_model_url(url):
    pattern = r"^https://huggingface.co/([^/]+)/([^/]+)$"
    match = re.match(pattern, url)
    return match.groups() if match else None

@backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, Exception), max_tries=20)
def make_http_request(url):
    response = requests.get(url)
    response.raise_for_status()
    return response

def smart_paginate(base_url, max_pages=100):
    """Smart pagination with backwards check if max_pages reached."""
    all_links = []
    page_num = 0
    truncated = False

    while page_num < max_pages:
        search_url = f"{base_url}&p={page_num}"
        response = make_http_request(search_url)
        soup = BeautifulSoup(response.text, "html.parser")
        model_divs = soup.find_all("div", class_="w-full truncate")

        if not model_divs and page_num < max_pages - 1:
            break

        for div in model_divs:
            header = div.find("header")
            if header:
                model_link = header.get("title")
                if model_link:
                    all_links.append(f"https://huggingface.co/{model_link}")

        page_num += 1

    if page_num >= max_pages:
        truncated = True
        print(f"Warning: Reached page {max_pages}. Results may be truncated.")

    return all_links, truncated

# ------------- Category Scrapers -------------

def get_finetuned_models_page(model_org, model_name, max_pages=100):
    return smart_paginate(f"https://huggingface.co/models?other=base_model:finetune:{model_org}/{model_name}", max_pages)

def get_adapter_models_page(model_org, model_name, max_pages=100):
    return smart_paginate(f"https://huggingface.co/models?other=base_model:adapter:{model_org}/{model_name}", max_pages)

def get_quantized_models_page(model_org, model_name, max_pages=100):
    return smart_paginate(f"https://huggingface.co/models?other=base_model:quantized:{model_org}/{model_name}", max_pages)

def get_merged_models_page(model_org, model_name, max_pages=100):
    return smart_paginate(f"https://huggingface.co/models?other=base_model:merge:{model_org}/{model_name}", max_pages)

@backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, Exception), max_tries=20)
def get_spaces_using_model(model_id, max_pages=100):
    try:
        url = f"https://huggingface.co/{model_id}"
        spaces = set()
        page = 0
        while page < max_pages:
            paginated_url = f"{url}/spaces?p={page}"
            page_response = requests.get(paginated_url)
            page_soup = BeautifulSoup(page_response.text, 'html.parser')
            new_spaces = set()
            for a_tag in page_soup.find_all('a', href=True):
                if '/spaces/' in a_tag['href']:
                    space_id = a_tag['href'].split('/spaces/')[-1]
                    space_name = a_tag.find('div', class_='truncate')
                    if space_name:
                        new_spaces.add(space_name.text.strip())
                    else:
                        new_spaces.add(space_id)
            if not new_spaces:
                break
            spaces.update(new_spaces)
            page += 1
        return {"visible_spaces": sorted(spaces), "total_count": len(spaces)}
    except Exception as e:
        print(f"Error fetching spaces: {e}")
        return {"visible_spaces": [], "total_count": 0}

def get_model_card(model_id):
    try:
        readme_path = hf_hub_download(repo_id=model_id, filename='README.md')
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Could not fetch model card: {e}")
        return "N/A"

def get_model_metadata(model_id):
    try:
        return api.model_info(model_id).__dict__
    except Exception as e:
        print(f"Could not fetch model metadata: {e}")
        return "N/A"

# ------------- BFS Finetune Explorer -------------

def bfs_finetunes(model_url, max_depth=3, max_pages=100):
    """Breadth-first crawl starting from a root model."""
    model_url = model_url.strip().rstrip('/')
    visited = set([model_url])
    results = []
    queue = deque([(model_url, 0)])

    while queue:
        current_url, depth = queue.popleft()
        if depth > max_depth:
            continue

        validated = validate_hf_model_url(current_url)
        if not validated:
            print(f"Invalid URL: {current_url}")
            continue

        model_org, model_name = validated
        model_id = f"{model_org}/{model_name}"
        print(f"\n{'  ' * depth}Fetching metadata for: {model_id}")

        try:
            model_metadata = get_model_metadata(model_id)
            json_metadata = json.dumps(model_metadata, default=str) if model_metadata != "N/A" else "N/A"
            model_card = get_model_card(model_id)

            finetune_links, truncated_finetunes = get_finetuned_models_page(model_org, model_name, max_pages)
            adapter_links, truncated_adapters = get_adapter_models_page(model_org, model_name, max_pages)
            quantized_links, truncated_quantized = get_quantized_models_page(model_org, model_name, max_pages)
            merges_links, truncated_merges = get_merged_models_page(model_org, model_name, max_pages)
            spaces_data = get_spaces_using_model(model_id, max_pages)

            any_truncation = truncated_finetunes or truncated_adapters or truncated_quantized or truncated_merges

            results.append({
                "model_id": model_id,
                "card": model_card,
                "metadata": json_metadata,
                "depth": depth,
                "children": finetune_links,
                "children_count": len(finetune_links),
                "adapters": adapter_links,
                "adapters_count": len(adapter_links),
                "quantized": quantized_links,
                "quantized_count": len(quantized_links),
                "merges": merges_links,
                "merges_count": len(merges_links),
                "spaces": spaces_data.get("visible_spaces", []),
                "spaces_count": spaces_data.get("total_count", 0),
                "truncated": any_truncation
            })

            for link in finetune_links:
                link = link.strip().rstrip('/')
                if link not in visited:
                    visited.add(link)
                    queue.append((link, depth + 1))

        except Exception as e:
            print(f"Error processing {model_id}: {e}")

    return results

# ------------- Save Functions -------------

def save_json(results, model_name, output_dir, is_truncated):
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_truncated" if is_truncated else ""
    filename = os.path.join(output_dir, f"{model_name}_finetunes{suffix}_{timestamp}.json")
    with open(filename, "w") as f:
        json.dump({"models": results}, f, indent=4, default=str)
    print(f"Results saved to {filename}")

def save_csv(results, model_name, output_dir, is_truncated):
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_truncated" if is_truncated else ""
    filename = os.path.join(output_dir, f"{model_name}_finetunes{suffix}_{timestamp}.csv")
    csv_results = []
    for entry in results:
        csv_entry = {k: ", ".join(v) if isinstance(v, list) else v for k, v in entry.items()}
        csv_results.append(csv_entry)
    df = pd.DataFrame(csv_results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# ------------- Main Runner -------------

def process_models_from_csv(csv_path, output_dir="output_models", max_depth=3, max_pages=100):
    df = pd.read_csv(csv_path)
    if 'id' not in df.columns:
        raise ValueError("CSV must contain an 'id' column.")

    for model_id in df['id']:
        model_url = f"https://huggingface.co/{model_id.strip()}"
        results = bfs_finetunes(model_url, max_depth, max_pages)

        if results:
            model_name = results[0]["model_id"].split("/")[-1]
            any_truncation = any(entry.get("truncated", False) for entry in results)
            print(f"\nSaving model: {model_name} (truncated={any_truncation})")

            save_json(results, model_name, output_dir, any_truncation)
            save_csv(results, model_name, output_dir, any_truncation)
        else:
            print(f"No results found for {model_id}.")

# ------------- CLI -------------

if __name__ == "__main__":
    csv_path = input("Enter path to CSV with model IDs: ").strip()
    output_dir = input("Enter output directory (default 'output_models'): ").strip() or "output_models"
    max_depth = int(input("Enter maximum search depth (default 3): ") or 3)
    max_pages = int(input("Enter maximum pages per search (default 100): ") or 100)

    process_models_from_csv(csv_path, output_dir, max_depth, max_pages)
