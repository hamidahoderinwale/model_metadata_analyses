from huggingface_hub import hf_hub_download
import os

repo_id = "midah/enriched_model_atlas_data"
chunk_files = [f"chunk_{i:04d}.jsonl" for i in range(NUM_CHUNKS)]  # Replace NUM_CHUNKS

with open("joined_dataset.jsonl", "w") as outfile:
    for fname in chunk_files:
        local_path = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
        with open(local_path, "r") as infile:
            for line in infile:
                outfile.write(line)
              
