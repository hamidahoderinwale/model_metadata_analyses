from huggingface_hub import HfApi
from bs4 import BeautifulSoup
import pandas as pd
import os
import datetime
import requests

# Setup
api = HfApi()
limit = 1723818
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = f"model_scan_{timestamp}"
os.makedirs(base_dir, exist_ok=True)

# Initialize counters
gated_count = 0
ungated_count = 0
all_model_records = []
chunk_index = 0
batch = []

# Paginated generator
model_generator = api.list_models(full=True, limit=limit)

try:
    for i, model in enumerate(model_generator, 1):
        is_gated = getattr(model, "gated", False)
        batch.append({
            "model_id": model.id,
            "gated": is_gated
        })

        if is_gated:
            gated_count += 1
        else:
            ungated_count += 1

        # Every `limit` models, save a chunk
        if i % limit == 0:
            df = pd.DataFrame(batch)
            df.to_csv(f"{base_dir}/chunk_{chunk_index:04d}.csv", index=False)
            print(f"Saved chunk {chunk_index:04d} with {len(batch)} models")
            all_model_records.extend(batch)
            batch = []
            chunk_index += 1

    # Save remaining models if total not divisible by limit
    if batch:
        df = pd.DataFrame(batch)
        df.to_csv(f"{base_dir}/chunk_{chunk_index:04d}.csv", index=False)
        print(f"Saved final chunk {chunk_index:04d} with {len(batch)} models")
        all_model_records.extend(batch)

except KeyboardInterrupt:
    print("\nInterrupted. Saving current state...")

# Save summary
df_all = pd.DataFrame(all_model_records)
total_on_hub = get_total_model_count()
summary_df = pd.DataFrame([{
    "timestamp": timestamp,
    "total_models_scanned": len(df_all),
    "gated_models": gated_count,
    "ungated_models": ungated_count,
    "total_models_on_huggingface": total_on_hub
}])

df_all.to_csv(f"{base_dir}/all_models.csv", index=False)
summary_df.to_csv(f"{base_dir}/summary.csv", index=False)

print(f"\nTotal models scanned: {len(df_all)}")
print(f"Gated: {gated_count} | Ungated: {ungated_count}")
print(f"All files saved to: {base_dir}/")
