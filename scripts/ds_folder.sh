#!/bin/bash
for json in ./*.json*; do
    # Skip if no JSON files are found
    [[ -e "$json" ]] || { echo "No .json files found."; break; }

    # Get the base filename without path
    filename="${json##*/}"
    
    # Remove all extensions
    base="${filename%%.*}"
    
    # Clean up the name by removing duplicate portions
    # This looks for patterns like "X_Y_X_Y" and converts to "X_Y"
    clean_name=$(echo "$base" | awk -F'_' '{
        len = NF/2;
        duplicate = 1;
        for (i=1; i<=len; i++) {
            if ($i != $(i+len)) {
                duplicate = 0;
                break;
            }
        }
        if (duplicate && len>0) {
            for (i=1; i<=len; i++) printf "%s%s", $i, (i<len?"_":"");
        } else {
            print $0;
        }
    }')
    
    # Find matching CSV (any file that starts with original base and has .csv)
    matching_csv=$(ls "./${base}".csv* 2>/dev/null | head -n 1)
    
    if [[ -f "$matching_csv" ]]; then
        mkdir -p "$clean_name"
        
        # Clean up the CSV filename too
        csv_filename="${matching_csv##*/}"
        clean_csv_name="${clean_name}.${matching_csv##*.}"
        
        mv -v "$json" "$clean_name/${clean_name}.${json##*.}"
        mv -v "$matching_csv" "$clean_name/$clean_csv_name"
    else
        echo "Skipping $json (no matching CSV for base: $base)"
    fi
done