# Call HF Hub API to get top models for tree generation and classify them as gated or ungated.

from huggingface_hub import HfApi
from bs4 import BeautifulSoup
import pandas as pd
import os
import datetime
import requests

# --- Setup ---
# Initialize Hugging Face API client and prepare output directory with timestamped folder
api = HfApi()
limit = 1723818  # Set upper limit for models to fetch (max possible on the Hub)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = f"model_scan_{timestamp}"
os.makedirs(base_dir, exist_ok=True)

# --- Initialize tracking variables ---
gated_count = 0
ungated_count = 0
all_model_records = []  # List to store all scanned models
chunk_index = 0
batch = []  # Temporary batch for chunked CSV saving

# --- Paginated model listing generator from HF Hub ---
model_generator = api.list_models(full=True, limit=limit)

try:
    # --- Iterate over all returned models ---
    for i, model in enumerate(model_generator, 1):
        is_gated = getattr(model, "gated", False)
        
        # Append model record to current batch
        batch.append({
            "model_id": model.id,
            "gated": is_gated
        })

        # Update counters
        if is_gated:
            gated_count += 1
        else:
            ungated_count += 1

        # Save each full batch as a chunked CSV
        if i % limit == 0:
            df = pd.DataFrame(batch)
            df.to_csv(f"{base_dir}/chunk_{chunk_index:04d}.csv", index=False)
            print(f"Saved chunk {chunk_index:04d} with {len(batch)} models")
            all_model_records.extend(batch)
            batch = []
            chunk_index += 1

    # --- Save any remaining models that didn't fill a full batch ---
    if batch:
        df = pd.DataFrame(batch)
        df.to_csv(f"{base_dir}/chunk_{chunk_index:04d}.csv", index=False)
        print(f"Saved final chunk {chunk_index:04d} with {len(batch)} models")
        all_model_records.extend(batch)

except KeyboardInterrupt:
    print("\nInterrupted. Saving current state...")

# --- Save summary CSVs with all model records and metadata ---
df_all = pd.DataFrame(all_model_records)

# NOTE: Ensure this function exists or define it. Otherwise, remove or mock:
# total_on_hub = get_total_model_count()
# For now, we'll comment it out to avoid an error.
# Replace with an actual API query if needed.

# Save summary statistics about the scan
summary_df = pd.DataFrame([{
    "timestamp": timestamp,
    "total_models_scanned": len(df_all),
    "gated_models": gated_count,
    "ungated_models": ungated_count,
    # "total_models_on_huggingface": total_on_hub  # comment if get_total_model_count is undefined
}])

# Output full list and summary stats
df_all.to_csv(f"{base_dir}/all_models.csv", index=False)
summary_df.to_csv(f"{base_dir}/summary.csv", index=False)

# --- Final logs ---
print(f"\nTotal models scanned: {len(df_all)}")
print(f"Gated: {gated_count} | Ungated: {ungated_count}")
print(f"All files saved to: {base_dir}/")
