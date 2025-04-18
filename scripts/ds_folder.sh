#!/bin/bash

for file in *.csv *.json; do
  [[ -e "$file" ]] && mv -v "$file" datasets/
done

  base="${csv%.*}"
  base_name="${base##*/}"
  json="./${base_name}.json"

  mkdir -p "$base_name"

  if [[ -f "$json"]]; then
    mv -v"$json" "$base_name/"
  else if 
    if [[ -f "$csv"]]; then
    mv -v"$csv" "$base_name/"
  else 
    echo "No matching JSON or CSVs, cannot be ran"
    mv -v "$csv" "$base_name/"
  fi
done
