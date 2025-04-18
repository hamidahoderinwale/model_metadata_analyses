#!/bin/bash

cd datasets || exit

for file in *.csv *.json; do
  [[ -e "$file" ]] && mv -v "$file" datasets/
done

for csv in ./*.csv; do
  [[ -e "$csv" ]] || { echo "No .csv files found."; break; }

  base="${csv%.*}"
  base_name="${base##*/}"
  json="./${base_name}.json"

  mkdir -p "$base_name"

  if [[ -f "$json" ]]; then
    mv -v "$csv" "$json" "$base_name/"
  else
    echo "No matching JSON for $csv, moving CSV alone."
    mv -v "$csv" "$base_name/"
  fi
done
