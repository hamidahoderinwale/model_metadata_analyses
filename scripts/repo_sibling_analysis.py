import csv
import os
import json
import backoff
from huggingface_hub import HfApi
from datasets import load_dataset
import matplotlib.pyplot as plt

# === Settings ===
INPUT_HF_DATASET = "Eliahu/ModelAtlasData"
CSV_PATH = "safetensors_index_presence.csv"
MD_PATH = "safetensors_summary.md"
PLOT_PATH = "safetensors_index_plot.png"

# === Setup ===
api = HfApi()

@backoff.on_exception(backoff.expo, (Exception,), max_tries=5)
def has_safetensors_index(model_id: str) -> bool:
    try:
        files = api.list_repo_files(model_id)
        return "model.safetensors.index.json" in files
    except Exception as e:
        print(f"⚠️ Error checking {model_id}: {e}")
        return False

def get_model_ids():
    print(f"📦 Streaming model IDs from {INPUT_HF_DATASET}...")
    ds = load_dataset(INPUT_HF_DATASET, split="train", streaming=True)
    return [item["id"] for item in ds if "id" in item]

def write_csv(results):
    print(f"📝 Writing CSV to {CSV_PATH}...")
    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id", "has_safetensors_index"])
        for model_id, has_index in results:
            writer.writerow([model_id, has_index])

def write_markdown_summary(results):
    total = len(results)
    num_with = sum(1 for _, has in results if has)
    num_without = total - num_with
    pct = (num_with / total) * 100 if total > 0 else 0

    with open(MD_PATH, "w") as f:
        f.write("# Safetensors Index Summary\n\n")
        f.write(f"**Total models checked:** {total}\n\n")
        f.write(f"- ✅ With `model.safetensors.index.json`: **{num_with}** ({pct:.1f}%)\n")
        f.write(f"- ❌ Without: **{num_without}** ({100 - pct:.1f}%)\n\n")
        f.write("## Chart\n")
        f.write(f"![safetensors chart]({PLOT_PATH})\n")

def generate_plot(results):
    print(f"📊 Generating plot at {PLOT_PATH}...")
    labels = ["With Index", "Without Index"]
    counts = [
        sum(1 for _, has in results if has),
        sum(1 for _, has in results if not has),
    ]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts)
    plt.title("Presence of model.safetensors.index.json")
    plt.ylabel("Number of Models")
    plt.savefig(PLOT_PATH)
    plt.close()

def main():
    print("🚀 Starting safetensors index scan...")
    model_ids = get_model_ids()

    results = []
    for i, model_id in enumerate(model_ids, 1):
        has_index = has_safetensors_index(model_id)
        results.append((model_id, has_index))
        if i % 100 == 0:
            print(f"  → Processed {i} models")

    write_csv(results)
    generate_plot(results)
    write_markdown_summary(results)

    print("✅ Done.")
    print(f"  • CSV: {CSV_PATH}")
    print(f"  • Markdown: {MD_PATH}")
    print(f"  • Plot: {PLOT_PATH}")

if __name__ == "__main__":
    main()
