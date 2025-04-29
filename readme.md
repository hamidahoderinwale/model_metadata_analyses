
# Hugging Face Model Tree Scraping

Notebook scripts to do the following using the [HF Hub API](https://huggingface.co/docs/huggingface.js/en/hub/README):
- Build complete model trees by following parent_model links
- Get metadata for a given model and it's fine-tunes
- Upload csv/json to `datasets` folder and write ` bash ./scripts/ds_folder.sh` to organize in model-specific folder

Once you've made a dataset, you could look into querying the data and try to:

- Get all models with base_model = "llama-2-70b" to see the whole family
- Follow parent_model links to trace specific lineages
- Group by base_model and count derivatives to measure influence

## Environment Setup

Before setting up the environment, ensure you have:
- Python 3.8 or higher installed
- pip package manager (usually included with Python)
Install the required dependencies:
- `pip install -r requirements.txt`
### Updating Requirements

If you're adding this project to an existing environment or need to update the requirements:

1. First activate your virtual environment:
   ```bash
   # For venv:
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate  # Windows

   # For Conda:
   conda activate your_env_name

2. Then generate an updated requirements file:
 `pip freeze > requirements.txt`

  [OneDrive](https://mcgill-my.sharepoint.com/:f:/g/personal/hamidah_oderinwale_mail_mcgill_ca/EjDp-Eo4PGdKtLxK3H84MHAB8TF1fwv0g5PTZnGu3JBa5Q?e=GaCW1Q)

## [Dataset Card](https://sites.research.google/datacardsplaybook/) [TK]

## [Serialize w/ Parquet](https://parquet.apache.org/)
