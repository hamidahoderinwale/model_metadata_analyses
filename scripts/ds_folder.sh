#!/bin/bash

for file in *.csv *.json; do
  [[ -e "$file" ]] && mv -v "$file" datasets/
done

cd datasets || exit 1

  base="${csv%.*}" 
  base="${json%.*}"
  base_name="${base##*/}"
  json="./${base_name}.json"
  csv="./${base_name}.csv"
  
mkdir -p "$base_name"

  if [[ -f "$json"]]; then
    mv -v"$json" "$base_name/"
  elif
    if [[ -f "$csv"]]; then
    mv -v"$csv" "$base_name/"
  elif 
    echo "No matching JSON or CSVs, cannot be ran"
    mv -v "$csv" "$base_name/"
  fi
done
