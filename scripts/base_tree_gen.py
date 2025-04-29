# Original tree scraping script without page limit
# If number of models for a model category is >2970, the backoff script will cause the script to end
# It's strongly suggested to use `[final]_truncated_multiple.py`

import requests
import pandas as pd
import datetime
import json
import re
import backoff
from huggingface_hub import HfApi, hf_hub_download
from bs4 import BeautifulSoup

# Initialize API
api = HfApi()

def validate_hf_model_url(url):
    pattern = r"^https://huggingface.co/([\w\-]+)/([\w\-]+)$"
    match = re.match(pattern, url)
    return match.groups() if match else None

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=20,
    factor=2,
    jitter=backoff.full_jitter
)
def make_http_request(url):
    response = requests.get(url)
    response.raise_for_status()
    return response

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=20,
    factor=2,
    jitter=backoff.full_jitter
)
def get_finetuned_models_page(model_org, model_name):
    all_model_links = []
    page_num = 0
    while True:
        try:
            search_url = f"https://huggingface.co/models?other=base_model:finetune:{model_org}/{model_name}&p={page_num}"
            response = make_http_request(search_url)
            soup = BeautifulSoup(response.text, "html.parser")
            model_divs = soup.find_all("div", class_="w-full truncate")
            if not model_divs:
                break
            for div in model_divs:
                header = div.find("header")
                if header:
                    model_link = header.get("title")
                    if model_link:
                        all_model_links.append(f"https://huggingface.co/{model_link}")
            page_num += 1
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            break
    return all_model_links

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=20,
    factor=2,
    jitter=backoff.full_jitter
)
def get_model_card(model_id):
    try:
        readme_path = hf_hub_download(repo_id=model_id, filename='README.md')
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"  Could not download model card: {str(e)[:100]}...")
        return "N/A"

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=5,
    factor=2,
    jitter=backoff.full_jitter
)
def get_model_metadata(model_id):
    try:
        return api.model_info(model_id).__dict__
    except Exception as e:
        print(f"  Could not fetch model metadata: {str(e)[:100]}...")
        return "N/A"

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=20,
    factor=2,
    jitter=backoff.full_jitter
)
def get_adapter_models_page(model_org, model_name):
    all_adapter_links = []
    page_num = 0
    while True:
        try:
            search_url = f"https://huggingface.co/models?other=base_model:adapter:{model_org}/{model_name}&p={page_num}"
            response = make_http_request(search_url)
            soup = BeautifulSoup(response.text, "html.parser")
            model_divs = soup.find_all("div", class_="w-full truncate")
            if not model_divs:
                break
            for div in model_divs:
                header = div.find("header")
                if header:
                    model_link = header.get("title")
                    if model_link:
                        all_adapter_links.append(f"https://huggingface.co/{model_link}")
            page_num += 1
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            break
    return all_adapter_links

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=20,
    factor=2,
    jitter=backoff.full_jitter
)
def get_quantized_models_page(model_org, model_name):
    all_quantized_links = []
    page_num = 0
    while True:
        try:
            search_url = f"https://huggingface.co/models?other=base_model:quantized:{model_org}/{model_name}&p={page_num}"
            response = make_http_request(search_url)
            soup = BeautifulSoup(response.text, "html.parser")
            model_divs = soup.find_all("div", class_="w-full truncate")
            if not model_divs:
                break
            for div in model_divs:
                header = div.find("header")
                if header:
                    model_link = header.get("title")
                    if model_link:
                        all_quantized_links.append(f"https://huggingface.co/{model_link}")
            page_num += 1
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            break
    return all_quantized_links

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=20,
    factor=2,
    jitter=backoff.full_jitter
)
def get_merged_models_page(model_org, model_name):
    all_merges_links = []
    page_num = 0
    while True:
        try:
            search_url = f"https://huggingface.co/models?other=base_model:merge:{model_org}/{model_name}&p={page_num}"
            response = make_http_request(search_url)
            soup = BeautifulSoup(response.text, "html.parser")
            model_divs = soup.find_all("div", class_="w-full truncate")
            if not model_divs:
                break
            for div in model_divs:
                header = div.find("header")
                if header:
                    model_link = header.get("title")
                    if model_link:
                        all_merges_links.append(f"https://huggingface.co/{model_link}")
            page_num += 1
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            break
    return all_merges_links

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=20,
    factor=2,
    jitter=backoff.full_jitter
)
def get_spaces_using_model(model_id):
    try:
        url = f"https://huggingface.co/{model_id}"
        spaces = set()
        # First page
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for a_tag in soup.find_all('a', href=True):
            if '/spaces/' in a_tag['href']:
                space_id = a_tag['href'].split('/spaces/')[-1]
                space_name = a_tag.find('div', class_='truncate')
                if space_name:
                    spaces.add(space_name.text.strip())
                else:
                    spaces.add(space_id)
        for div in soup.find_all('div', class_='truncate group-hover:underline'):
            div_text = div.text.strip()
            if '/' in div_text:
                spaces.add(div_text)
        # Paginate
        page = 0
        while True:
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
            for div in page_soup.find_all('div', class_='truncate group-hover:underline'):
                div_text = div.text.strip()
                if '/' in div_text:
                    new_spaces.add(div_text)
            if not new_spaces:
                break
            spaces.update(new_spaces)
            page += 1
        return {
            "visible_spaces": sorted(spaces),
            "total_count": len(spaces)
        }
    except Exception as e:
        print(f"Error: {str(e)[:200]}...")
        return {"visible_spaces": [], "total_count": 0}

def dfs_finetunes(model_url, visited, depth=0, results=None):
    if results is None:
        results = []

    model_url = model_url.strip().rstrip('/')
    if model_url in visited:
        return results
    visited.add(model_url)

    validated = validate_hf_model_url(model_url)
    if not validated:
        print(f"Invalid URL: {model_url}")
        results.append({
            "model_id": model_url,
            "card": "N/A",
            "metadata": "N/A",
            "depth": depth,
            "children": [],
            "children_count": 0,
            "adapters": [],
            "adapters_count": 0,
            "quantized": [],
            "quantized_count": 0,
            "merges": [],
            "merges_count": 0,
            "spaces": [],
            "spaces_count": 0
        })
        return results

    model_org, model_name = validated
    model_id = f"{model_org}/{model_name}"
    print(f"\n{'  ' * depth}Fetching metadata for: {model_id}")

    try:
        # Get metadata
        model_metadata = get_model_metadata(model_id)
        json_metadata = json.dumps(model_metadata, default=str) if model_metadata != "N/A" else "N/A"
        # Get card
        model_card = get_model_card(model_id)
        # Get children and related models
        finetune_links = get_finetuned_models_page(model_org, model_name)
        print(f"{'  ' * depth}Found {len(finetune_links)} fine-tunes at depth {depth}.")
        adapter_links = get_adapter_models_page(model_org, model_name)
        print(f"{'  ' * depth}Found {len(adapter_links)} adapter models for {model_id}.")
        quantized_links = get_quantized_models_page(model_org, model_name)
        print(f"{'  ' * depth}Found {len(quantized_links)} quantized models for {model_id}.")
        merges_links = get_merged_models_page(model_org, model_name)
        print(f"{'  ' * depth}Found {len(merges_links)} merged models for {model_id}.")
        spaces_data = get_spaces_using_model(model_id)
        print(f"{'  ' * depth}Found {spaces_data['total_count']} spaces for {model_id}.")

        results.append({
            "model_id": model_id,
            "card": model_card if model_card else "N/A",
            "metadata": json_metadata if json_metadata else "N/A",
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
            "spaces_count": spaces_data.get("total_count", 0)
        })

        for link in finetune_links:
            dfs_finetunes(link.strip().rstrip('/'), visited, depth + 1, results)
        return results

    except Exception as e:
        print(f"Error processing {model_id}: {str(e)}")
        results.append({
            "model_id": model_id,
            "card": "N/A",
            "metadata": "N/A",
            "depth": depth,
            "children": [],
            "children_count": 0,
            "adapters": [],
            "adapters_count": 0,
            "quantized": [],
            "quantized_count": 0,
            "merges": [],
            "merges_count": 0,
            "spaces": [],
            "spaces_count": 0
        })
        return results

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json(results, model_name):
    filename = f"{model_name}_finetunes_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({"models": results}, f, indent=4, default=str)
    print(f"Results saved to {filename}")

def save_csv(results, model_name):
    filename = f"{model_name}_finetunes_{timestamp}.csv"
    csv_results = []
    for entry in results:
        csv_entry = {k: ", ".join(v) if isinstance(v, list) else v for k, v in entry.items()}
        csv_results.append(csv_entry)
    df = pd.DataFrame(csv_results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    model_url = input("Enter the Hugging Face model URL: ").strip()
    visited = set()
    results = dfs_finetunes(model_url, visited)

    if results:
        model_name = results[0]["model_id"].split("/")[-1]
        print(f"\nTotal models found (all depths): {len(results)}")
        save_json(results, model_name)
        save_csv(results, model_name)
    else:
        print("No fine-tuned models found.")
