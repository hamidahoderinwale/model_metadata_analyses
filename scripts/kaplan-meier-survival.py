# In-progress 
# Plot 1: Kaplan–Meier survival curve showing how long models persist through generational depth,
# split by whether they include adapters. Models with adapters generally survive (branch) deeper,
# indicating greater reuse or longevity in the ecosystem.

#!/usr/bin/env python3
!pip install lifelines
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from scipy.stats import linregress

# --- Configuration ---
INPUT_CSV = "joined_output.csv"
OUTPUT_DIR = "ecology_analysis_pngs"
MISSING_CARDS_CSV = "missing_model_cards.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Utility Functions ---
def gini(array):
    """Compute Gini coefficient of array."""
    arr = np.array(array, dtype=float).flatten()
    if arr.min() < 0:
        arr -= arr.min()
    arr += 1e-9
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n - 1) * arr)) / (n * arr.sum()))

def shannon_diversity(counts):
    """Compute Shannon diversity index."""
    props = counts / counts.sum()
    return -np.sum(props * np.log2(props + 1e-9))

# --- Load & preprocess ---
df = pd.read_csv(INPUT_CSV, low_memory=False)
# parse children lists from string -> list
df['children'] = (
    df.get('children', '[]')
      .fillna('[]')
      .apply(lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith('[') else [])
)
# ensure numeric
for col in ['children_count','adapters_count','quantized_count']:
    df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0).astype(int)
df['base_model'] = df.loc[df['depth']==0, 'model_id']  # depth=0 are base

# --- 1. Modular innovation: survival curve w/ vs w/o adapters ---
kmf = KaplanMeierFitter()
plt.figure(figsize=(8,6))
for has_adapter, grp in df.groupby(df['adapters_count']>0):
    durations = grp['depth']  # treat depth as “lifetime” proxy
    event_observed = np.ones_like(durations)  # everyone eventually “dies” (leaf)
    label = "with adapters" if has_adapter else "without adapters"
    kmf.fit(durations, event_observed, label=label)
    kmf.plot_survival_function(ci_show=False)
plt.title("Kaplan–Meier Survival by Adapter Presence")
plt.xlabel("Tree Depth (generations)")
plt.ylabel("Survival (model being non‐leaf)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"km_survival_adapters.png"))
plt.close()

# --- 2. Ecosystem saturation: base model count vs avg descendants ---
base_desc = df[df['depth']==0].copy()
# map each base to its total descendants
desc_map = {}
for _, row in df[df['depth']==0].iterrows():
    # count all nodes in its weakly connected component minus 1
    comp = df.loc[df['model_id']==row['model_id'], 'children_count'].sum()
    desc_map[row['model_id']] = comp
base_desc['avg_descendants'] = base_desc['model_id'].map(desc_map)
base_desc['generation'] = range(len(base_desc))  # increasing base model count
plt.figure(figsize=(8,6))
plt.scatter(base_desc.index+1, base_desc['avg_descendants'], alpha=0.6)
slope, intercept, _, _, _ = linregress(base_desc.index+1, base_desc['avg_descendants'])
x = np.array([1, len(base_desc)])
plt.plot(x, slope*x+intercept, 'r--', label=f"fit (slope={slope:.2f})")
plt.title("Ecosystem Saturation: Base Models vs Avg Descendants")
plt.xlabel("Cumulative Number of Base Models")
plt.ylabel("Average Descendant Count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"ecosystem_saturation.png"))
plt.close()

# --- 3. Lineage distribution: heavy‐tailed & Gini ---
all_desc = df['children_count']
plt.figure(figsize=(8,4))
plt.hist(all_desc, bins=50, log=True, edgecolor='k')
plt.title("Histogram of Descendant Counts (log y‐axis)")
plt.xlabel("Number of Children")
plt.ylabel("Frequency (log)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"descendant_histogram.png"))
plt.close()

gini_coeff = gini(all_desc)
with open(os.path.join(OUTPUT_DIR,"gini_descendants.txt"), 'w') as f:
    f.write(f"Gini coefficient of descendant counts: {gini_coeff:.4f}\n")

# --- 4. Adapters as “mutations”: diversity pre/post ---
# assume 'tags' column is JSON list
df['tags'] = df['tags'].apply(lambda x: json.loads(x) if isinstance(x,str) and x.startswith('[') else [])
def diversity_by_group(mask, name):
    sub = df[mask]
    # count how many different pipeline_tags or other tag categories
    flat = pd.Series([t for tags in sub['tags'] for t in tags])
    counts = flat.value_counts()
    return shannon_diversity(counts.values)
div_no = diversity_by_group(df['adapters_count']==0, "no adapters")
div_yes = diversity_by_group(df['adapters_count']>0, "with adapters")
with open(os.path.join(OUTPUT_DIR,"diversity_adapters.txt"), 'w') as f:
    f.write(f"Shannon diversity without adapters: {div_no:.4f}\n")
    f.write(f"Shannon diversity with adapters: {div_yes:.4f}\n")

# --- 5. Quantization & specialization: compare pipeline_tag spread ---
quant = df[df['quantized_count']>0]
nonq = df[df['quantized_count']==0]
def tag_distribution(sub):
    flat = [t for tags in sub['tags'] for t in tags if not t.startswith('base_model')]
    return pd.Series(flat).value_counts().head(10)
dist_q = tag_distribution(quant)
dist_nq = tag_distribution(nonq)
pd.DataFrame({'quantized': dist_q, 'non_quantized': dist_nq}).plot.barh(figsize=(8,6))
plt.title("Top Tags: Quantized vs Non-Quantized Models")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"quantization_tag_comparison.png"))
plt.close()

# --- 6. Missing model cards CSV ---
missing = df[df['card'].isna() | df['card'].isin(['', 'N/A'])]
missing[['model_id','depth','children_count','adapters_count','quantized_count']].to_csv(
    MISSING_CARDS_CSV, index=False
)

print("All analyses complete.")
print(f"– {MISSING_CARDS_CSV} written with {len(missing)} missing‐card models.")```

'''**What this does**
1. **Kaplan–Meier**: compares base‐to‐leaf survival curves for models *with* vs *without* adapters.
2. **Ecosystem saturation**: scatter of (cumulative base models) vs (their avg descendants) with a regression line.
3. **Heavy‐tailed lineage**: histogram of children counts on a log scale plus Gini coefficient.
4. **Adapter “mutation” diversity**: Shannon diversity of tags pre/post adapters.
5. **Quantization & Task Specialization**: bar chart comparing the top 10 tags in quantized vs non-quantized models.
6. **Missing cards**: emits a CSV of all models whose `card` column is empty, null, or “N/A”.'''
