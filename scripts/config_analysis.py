import os
import pandas as pd
import json
import re
from collections import defaultdict, Counter
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# === SETTINGS ===
INPUT_CSV = "models_metadata.csv"  # must contain at least `model_id` column
OUTPUT_CSV = "model_architecture_analysis.csv"

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV)

# Optional: check for availability of model.safetensors.index.json (if not already filtered)
# This assumes you have pre-filtered for models with index files if needed
model_ids = df['model_id'].dropna().unique().tolist()

# === RESULTS ===
results = []

def extract_languages_from_config(config):
    lang_field = config.get("language", [])
    if isinstance(lang_field, list):
        return lang_field
    elif isinstance(lang_field, str):
        return [lang_field]
    return []

for model_id in tqdm(model_ids, desc="Processing models"):
    try:
        # Download config.json
        config_path = hf_hub_download(repo_id=model_id, filename="config.json", force_download=False)
        with open(config_path, "r") as f:
            config = json.load(f)

        # Download model.safetensors.index.json
        index_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json", force_download=False)
        with open(index_path, "r") as f:
            index = json.load(f)

        # === ARCHITECTURE ANALYSIS ===
        layer_prefix = "model.layers."
        layer_types = defaultdict(int)
        layer_ids = set()

        for param_name in index.get("weight_map", {}):
            if param_name.startswith(layer_prefix):
                parts = param_name.split(".")
                if len(parts) > 3:
                    layer_id = parts[2]
                    layer_type = parts[3]
                    layer_types[layer_type] += 1
                    layer_ids.add(layer_id)

        layer_count = config.get("num_hidden_layers", len(layer_ids))
        vocab_size = config.get("vocab_size", "Unknown")
        qcfg = config.get("quantization_config", {})

        # Shape stats
        shape_counts = {}
        if "shapes" in index:
            shape_counts = Counter(tuple(v) for v in index["shapes"].values())

        # Language info
        languages = extract_languages_from_config(config)

        results.append({
            "model_id": model_id,
            "layer_count": layer_count,
            "layer_types": dict(layer_types),
            "vocab_size": vocab_size,
            "quantization_method": qcfg.get("quant_method", ""),
            "quant_format": qcfg.get("fmt", ""),
            "quant_weight_block_size": qcfg.get("weight_block_size", ""),
            "quant_activation_scheme": qcfg.get("activation_scheme", ""),
            "language_codes": languages,
            "num_unique_shapes": len(shape_counts),
            "shape_counts": dict(shape_counts)
        })

    except Exception as e:
        print(f"⚠️ Skipped {model_id} due to error: {e}")
        continue

# === SAVE RESULTS ===
df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Analysis complete. Results saved to {OUTPUT_CSV}.")

# === LANGUAGE DISTRIBUTION SUMMARY ===
all_langs = [lang for row in df_out['language_codes'].dropna() for lang in row]
lang_counter = Counter(all_langs)
print("\n🌍 Language distribution across models:")
for lang, count in lang_counter.most_common():
    print(f"{lang}: {count}")

# Optional: Save language distribution
lang_df = pd.DataFrame(lang_counter.items(), columns=["language", "count"]).sort_values("count", ascending=False)
lang_df.to_csv("language_distribution_from_config.csv", index=False)
