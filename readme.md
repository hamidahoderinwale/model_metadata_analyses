
# Dataset Schema Design 

** Note: Below is partially-generated (by Claude) summary of the dataset schema structure.

This schema is specifically designed for LLM-assisted data collection. Here are the key features:

This approach allows you to:

- Build complete model trees by following parent_model links
- Group models by their base_model for family analysis
- Here, I would imagine that if the nested data in the schema were flattened: you would extract each property into its own column in a table

You could query the data in multiple ways:

- Get all models with base_model = "llama-2-70b" to see the whole family
- Follow parent_model links to trace specific lineages
- Group by base_model and count derivatives to measure influence

The schema is flexible enough to handle complex cases like:

- Models with multiple parent influences

You can validate the dataset schema as updates are made by using the [jsonlint](https://github.com/zaach/jsonlint) package in your terminal (which you must download first `npm install jsonlint -g`). Then, typing `jsonlint <name of your JSON file>`.

## External Links
- GColab visualizations: https://colab.research.google.com/drive/1Y-9HkUkCbAJf4njmVy-x1zfApOoOV25D?usp=sharing
- Libraries Dataset (WIP): https://docs.google.com/spreadsheets/d/1tI67UpqRyy7IEZNlBvIKanvqw5trYUxMnLNtufRQ7o8/edit?usp=sharing
