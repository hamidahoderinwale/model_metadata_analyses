
# Hugging Face Model Hub meta-analysis repo

Notebook scripts to do the following using the [HF Hub API](https://huggingface.co/docs/huggingface.js/en/hub/README):
- Build complete model trees by following parent_model links
- Get metadata for a given model and it's fine-tunes
- Upload csv/json to `datasets` folder and write ` bash ./scripts/ds_folder.sh` to organize in model-specific folder

Once you've made a dataset, you could look into querying the data and try to:

- Get all models with base_model = "llama-2-70b" to see the whole family
- Follow parent_model links to trace specific lineages
- Group by base_model and count derivatives to measure influence


## [Dataset Card](https://sites.research.google/datacardsplaybook/) [TK]

## [Serialize w/ Parquet](https://parquet.apache.org/)
