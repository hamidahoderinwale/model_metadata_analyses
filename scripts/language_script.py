import pandas as pd
import re
import ast
from collections import Counter
from langcodes import Language

# === Settings ===
INPUT_CSV = "models_metadata.csv"  # Must include at least a 'card' column
OUTPUT_LANG_CSV = "language_distribution.csv"
OUTPUT_LANG_MD = "language_summary.md"

# === Function: Parse languages from model card text ===
def extract_languages(card):
    if pd.isna(card):
        return []
    
    match = re.search(r'language:\s*(\[[^\]]*\]|(?:\s*-\s*[a-z]{2}[\-\w]*\s*)+)', card, re.IGNORECASE)
    if not match:
        return []

    block = match.group(1).strip()
    if block.startswith("["):
        try:
            return ast.literal_eval(block)
        except:
            return []
    lines = block.split("\n")
    return [re.sub(r"^\s*-\s*", "", line).strip() for line in lines if line.strip().startswith("-")]

# === Function: Map language code to full name ===
def get_language_name(code):
    try:
        return Language.get(code).display_name()
    except:
        return code  # fallback to code if invalid

# === Load CSV ===
df = pd.read_csv(INPUT_CSV)
df['languages'] = df['card'].apply(extract_languages)

# === Count language codes ===
all_langs = [lang for row in df['languages'] if isinstance(row, list) for lang in row]
lang_counter = Counter(all_langs)

# === Save to CSV ===
lang_df = pd.DataFrame(lang_counter.items(), columns=["language_code", "count"])
lang_df["language_name"] = lang_df["language_code"].apply(get_language_name)
lang_df = lang_df.sort_values("count", ascending=False)
lang_df.to_csv(OUTPUT_LANG_CSV, index=False)

# === Generate Markdown ===
with open(OUTPUT_LANG_MD, "w") as f:
    f.write("# 🌍 Language Distribution from Model Cards\n\n")
    f.write("This table summarizes the most common human languages listed in Hugging Face model metadata.\n\n")
    f.write("| Language Code | Language Name | Model Count |\n")
    f.write("|---------------|----------------|--------------|\n")
    for _, row in lang_df.iterrows():
        f.write(f"| `{row['language_code']}` | {row['language_name']} | {row['count']} |\n")

print(f"✅ Saved: {OUTPUT_LANG_CSV}, {OUTPUT_LANG_MD}")
