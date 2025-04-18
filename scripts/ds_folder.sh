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

  [[ -f "${base}.csv" ]] && mv -v "${base}.csv" "$folder/"
  [[ -f "${base}.json" ]] && mv -v "${base}.json" "$folder/"
done 
