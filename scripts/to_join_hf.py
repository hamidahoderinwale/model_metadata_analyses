from datasets import load_dataset

ds = load_dataset("midah/enriched_model_atlas_data", split="train")

ds.to_csv("joined_model_atlas.csv")
print("Saved dataset as joined_model_atlas.csv")
