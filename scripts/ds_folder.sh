#!/bin/bash

cd datasets || exit

for json in ./*.json; do
  # Skip if no JSON files are found
  [[ -e "$json" ]] || { echo "No .json files found."; break; }

  base="${json%.*}"        # Remove extension
  base_name="${base##*/}"  # Remove path prefix
  csv="./${base_name}.csv" # Expected matching CSV file

  if [[ -f "$csv" ]]; then
    mkdir -p "$base_name"
    mv -v "$json" "$csv" "$base_name/"
  else
    echo "Skipping $json (no matching CSV: $csv)"
  fi
done
