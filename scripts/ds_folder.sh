#!/bin/bash

# Step 1: Move all .csv and .json files into datasets/
mkdir -p datasets
for file in *.csv *.json; do
  [[ -e "$file" ]] && mv -v "$file" datasets/
done

cd datasets || exit 1

# Step 3: Loop through all .csv and .json files
for file in *.csv *.json; do
  [[ -e "$file" ]] || continue
  base="${file%.*}"
  base_name="${base##*/}"
  csv="${base_name}.csv"
  json="${base_name}.json"

  mkdir -p "$base_name"

  if [[ -f "$csv" && -f "$json" ]]; then
    echo "Moving both CSV and JSON for $base_name"
    mv -v "$csv" "$json" "$base_name/"
  elif [[ -f "$csv" ]]; then
    echo "Only CSV exists for $base_name"
    mv -v "$csv" "$base_name/"
  elif [[ -f "$json" ]]; then
    echo "Only JSON exists for $base_name"
    mv -v "$json" "$base_name/"
  else
    echo "No matching files for $base_name"
  fi
done
