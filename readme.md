
# Hugging Face Model Hub Meta-analysis project

Notebook scripts to do the following:
- Build complete model trees by following parent_model links
- Get metadata for a given model and it's fine-tunes

You could query the data in multiple ways:

- Get all models with base_model = "llama-2-70b" to see the whole family
- Follow parent_model links to trace specific lineages
- Group by base_model and count derivatives to measure influence

You can validate the dataset schema as updates are made by using the [jsonlint](https://github.com/zaach/jsonlint) package in your terminal (which you must download first `npm install jsonlint -g`). Then, typing `jsonlint <name of your JSON file>`.

We've yet to look into fine-tunes, adapters, and model merges.

## External Links
- Scraping Notebook: https://colab.research.google.com/drive/1Rg2_wSjXzVJq-CMQZQU4GyMVBlZo0og3?usp=sharing
- GColab visualizations (Early/Ignore): https://colab.research.google.com/drive/1Y-9HkUkCbAJf4njmVy-x1zfApOoOV25D?usp=sharing
- Libraries Dataset (WIP): https://docs.google.com/spreadsheets/d/1tI67UpqRyy7IEZNlBvIKanvqw5trYUxMnLNtufRQ7o8/edit?usp=sharing
