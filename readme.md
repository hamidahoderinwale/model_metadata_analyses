
# Hugging Face Model Trees

This project provides Python scripts to systematically map and analyze Hugging Face model ecosystems by building complete family trees of models and their fine-tuned derivatives. 

Using the [Hugging Face Model Hub](https://huggingface.co/docs/hub/en/index), we recursively discover parent-child relationships between models, collect comprehensive metadata, and export the data in structured formats for further analysis.

This repo links to the resulting datasets and hosts the scripts used for our analyses. This dataset enables researchers to trace model lineages, measure the influence of base models, and understand the propagation patterns within the open-source AI model ecosystem.

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

## Links 
- [Hugging Face dataset of >1100 trees & 53,991 models from ordered list of most-liked models](https://huggingface.co/datasets/midah/removed_gemma_trees)
- [Hugging Face dataset of models in our analysis](https://huggingface.co/datasets/midah/ecosystem_map)
- [Hugging Face dataset of HF libraries and their user stats (in files)](https://huggingface.co/datasets/midah/hf_libraries_use)
- [Plots (private)](https://drive.google.com/drive/folders/1Z_K9Jw-MK-CutZ9trj21wdc_4HA9_wHA?usp=sharing)
- [Dataset Card (TK)](https://sites.research.google/datacardsplaybook/)
