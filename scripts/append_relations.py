!pip install datasets backoff pandas bs4 tqdm
from tqdm import tqdm
import requests
import backoff
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset
import pandas as pd

# get page 
@backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, Exception), max_tries=10)
def fetch_model_page(model_id):
    url = f"https://huggingface.co/{model_id}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ModelLineageBot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text

# parent lineage rows 
def extract_lineage_rows_from_html(html, model_row):
    soup = BeautifulSoup(html, "html.parser")
    model_input = model_row["model_id"]
    model_id = model_input.split("/")[-1]
    base_model_id = None
    lineage_rows = []

    base_model_section = soup.find("p", string=lambda s: s and s.strip().lower() == "base model")
    if not base_model_section:
        return []

    tree_container = base_model_section.find_parent("div", class_="text-smd")
    if not tree_container:
        return []

    # Extract base_model_id
    flex_div = base_model_section.find_parent("div", class_="flex")
    if flex_div:
        base_model_link = flex_div.find("a", href=True)
        if base_model_link:
            base_model_id = base_model_link.text.strip()

    rel_type_map = {
        "finetuned": "finetuned",
        "finetunes": "finetuned",
        "quantized": "quantized",
        "adapter": "adapter",
        "adapters": "adapter",
        "merge": "merged",
        "merges": "merged"
    }

    for flex_div in tqdm(tree_container.find_all("div", class_="flex", recursive=False), desc="finding base, parent, and parent relationship"):
        label_div = flex_div.find("div", class_="font-semibold")
        if not label_div:
            continue

        rel_type_raw = label_div.get_text(separator=" ", strip=True).lower()
        rel_type = next((rel_type_map[key] for key in rel_type_map if key in rel_type_raw), None)
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

            if parent_id == model_id:
                continue

            if "/" in parent_id:
                lineage_rows.append({
                    "model_id": model_id,
                    "gated": model_row.get("gated"),
                    "card": model_row.get("card"),
                    "metadata": model_row.get("metadata"),
                    "parent_model_id": parent_id,
                    "parent_relationship": rel_type,
                    "base_model_id": base_model_id
                })

    return lineage_rows

# for testing 
def process_first_10_models(input_dataset):
    all_rows = []
    for row in input_dataset.select(range(min(10, len(input_dataset)))):
        model_id = row["model_id"]
        try:
            html = fetch_model_page(model_id)
            lineage_rows = extract_lineage_rows_from_html(html, row)
            all_rows.extend(lineage_rows)
        except Exception as e:
            print(f"Error fetching {model_id}: {e}")
    return Dataset.from_pandas(pd.DataFrame(all_rows))

  # for prod
'''def process_all_models(input_dataset):
    all_rows = []
    for row in tqdm(input_dataset, desc="processing models"):
        model_id = row["model_id"]
        try:
            html = fetch_model_page(model_id)
            lineage_rows = extract_lineage_rows_from_html(html, row)
            all_rows.extend(lineage_rows)
        except Exception as e:
            print(f"Error fetching {model_id}: {e}")
    return Dataset.from_pandas(pd.DataFrame(all_rows))'''


if __name__ == "__main__":
    INPUT_DATASET = "midah/ecosystem_map"

    input_dataset = load_dataset(INPUT_DATASET,  data_files="all_models.csv", split="train")
    output_dataset = process_first_10_models(input_dataset)

    # Save locally instead of pushing for test
    output_dataset.to_csv("lineage_test_output.csv")
    # use to push to the hub 
    # 'output_dataset.push_to_hub("your_org/your_output_dataset_test")'
    print("appended dataset is complete")

    print("Test output saved to lineage_test_output.csv")
