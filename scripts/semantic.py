# tokenized model card data as json with dictionary and list of tokens with frequency
import numpy as np
import pandas as pd
import re
import json
import os
from collections import Counter

def tokenize_from_csv(file_path=None):
    if file_path is None:
        file_path = input("Enter the path to the CSV file: ")
        model_name = file_path.split(".csv")[0].split(".")[-1]


    # Read the CSV file
    df = pd.read_csv(file_path, encoding='utf-8')

    # Check if 'card' column exists
    if 'card' not in df.columns:
        raise KeyError("The CSV file does not contain a 'card' column")

    # Extract text from the 'card' column
    unstructured_cards = df['card'].tolist()

    # Initialize lists for tokens and a Counter for frequencies
    all_tokens = []

    # Process each card
    for card in unstructured_cards:
        # Convert to string in case it's not already
        text = str(card)

        # Clean the text (remove non-alphanumeric characters and convert to lowercase)
        cleaned_text = re.sub(r'\W+', ' ', text.lower())

        # Split into tokens
        tokens = cleaned_text.split()

        # Add to our overall token list
        all_tokens.extend(tokens)

    # Count token frequencies
    token_frequencies = Counter(all_tokens)

    # Create data to store
    data_to_store = {
        "tokens": all_tokens,
        "token_count": len(all_tokens),
        "unique_token_count": len(token_frequencies),
        "token_frequencies": {token: freq for token, freq in token_frequencies.items()}
    }

    # Save to JSON
    output_filename = f"{model_name}_tokenized_data.json"
    with open(output_filename, "w") as f:
        json.dump(data_to_store, f, indent=4)
        
    print(f"Processed {len(unstructured_cards)} cards with {len(all_tokens)} total tokens")
    print(f"Found {len(token_frequencies)} unique tokens")

    return all_tokens, token_frequencies

if __name__ == "__main__":
    tokens, frequencies = tokenize_from_csv()

    # Display the most common tokens
    print("\nMost common tokens:")
    for token, count in frequencies.most_common(10):
        print(f"{token}: {count}")
