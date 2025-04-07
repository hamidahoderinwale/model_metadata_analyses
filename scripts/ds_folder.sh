#!/bin/bash
cd datasets && for json in ./*.json; do 
  base="${json%.*}";
  base="${base##*/}";  # Remove path prefix if exists
  csv="./${base}.csv";
  if [[ -f "$csv" ]]; then
    mkdir -p "$base" && mv -v "$json" "$csv" "$base/";
  else
    echo "Skipping ./*.json (no matching CSV)";
  fi;
done
