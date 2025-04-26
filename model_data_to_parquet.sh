#!/bin/bash

# HuggingFace Model Data to Parquet Converter
# This script converts CSV and JSON model data files to Parquet format

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Check if required Python packages are installed
echo "Checking required Python packages..."
python3 -c "import pandas, pyarrow, json" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required Python packages..."
    pip install pandas pyarrow
fi

# Create a Python script for the conversion logic
cat > convert_to_parquet.py << 'EOF'
import os
import sys
import json
import pandas as pd
import glob
from pathlib import Path
import re

def is_json_file(filepath):
    """Check if a file is likely a JSON file based on content or extension."""
    # Check extension
    if filepath.lower().endswith('.json'):
        return True
    
    # Check content
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read(100).strip()  # Read first 100 chars
            if content.startswith('{') or content.startswith('['):
                return True
    except:
        pass
    
    return False

def is_csv_file(filepath):
    """Check if a file is likely a CSV file based on content or extension."""
    # Check extension
    if filepath.lower().endswith('.csv'):
        return True
    
    # Check content - look for comma-separated values in the first few lines
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for _ in range(3):  # Check first 3 lines
                line = f.readline().strip()
                if line and ',' in line and len(line.split(',')) > 1:
                    return True
    except:
        pass
    
    return False

def convert_json_to_parquet(json_file, output_dir):
    """Convert a JSON file to Parquet format."""
    print(f"Converting JSON file: {json_file}")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict) and 'models' in data:
            # JSON with 'models' key
            df = pd.DataFrame(data['models'])
        elif isinstance(data, list):
            # JSON with a list of records
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # JSON with a single record
            df = pd.DataFrame([data])
        else:
            print(f"  Warning: Unsupported JSON structure in {json_file}")
            return None
        
        # Create output filename
        filename = os.path.basename(json_file).split('.')[0] + '.parquet'
        output_path = os.path.join(output_dir, filename)
        
        # Save as Parquet
        df.to_parquet(output_path, index=False)
        print(f"  Saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"  Error converting {json_file}: {str(e)}")
        return None

def convert_csv_to_parquet(csv_file, output_dir):
    """Convert a CSV file to Parquet format."""
    print(f"Converting CSV file: {csv_file}")
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        
        # Create output filename
        filename = os.path.basename(csv_file).split('.')[0] + '.parquet'
        output_path = os.path.join(output_dir, filename)
        
        # Save as Parquet
        df.to_parquet(output_path, index=False)
        print(f"  Saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"  Error converting {csv_file}: {str(e)}")
        return None

def find_data_files(input_path):
    """Find all CSV and JSON files in the input path, including files without extensions."""
    data_files = {'csv': [], 'json': []}
    
    # If input path is a file, check it directly
    if os.path.isfile(input_path):
        if is_csv_file(input_path):
            data_files['csv'].append(input_path)
        elif is_json_file(input_path):
            data_files['json'].append(input_path)
        return data_files
    
    # If input path is a directory, walk through it
    for root, _, files in os.walk(input_path):
        for file in files:
            filepath = os.path.join(root, file)
            if is_csv_file(filepath):
                data_files['csv'].append(filepath)
            elif is_json_file(filepath):
                data_files['json'].append(filepath)
    
    return data_files

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_to_parquet.py <input_path> <output_dir>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all data files
    print(f"Searching for data files in {input_path}...")
    data_files = find_data_files(input_path)
    
    print(f"Found {len(data_files['csv'])} CSV files and {len(data_files['json'])} JSON files.")
    
    # Convert all files
    converted_files = []
    
    # Convert CSV files
    for csv_file in data_files['csv']:
        result = convert_csv_to_parquet(csv_file, output_dir)
        if result:
            converted_files.append(result)
    
    # Convert JSON files
    for json_file in data_files['json']:
        result = convert_json_to_parquet(json_file, output_dir)
        if result:
            converted_files.append(result)
    
    print(f"\nConversion complete. Converted {len(converted_files)} files to Parquet format.")
    
    # If no files were converted, exit with error
    if not converted_files:
        print("No files were converted. Check your input path and file formats.")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Function to show usage information
show_usage() {
    echo "Usage: $0 <input_path> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  input_path     Directory or file containing model data files (CSV/JSON)"
    echo "  output_dir     Directory where Parquet files will be saved"
    echo ""
    echo "Example:"
    echo "  $0 ./model_data ./parquet_output"
}

# Check if arguments are provided
if [ $# -lt 2 ]; then
    show_usage
    exit 1
fi

input_path="$1"
output_dir="$2"

# Check if input path exists
if [ ! -e "$input_path" ]; then
    echo "Error: Input path does not exist: $input_path"
    exit 1
fi

# Run the Python script
echo "Starting conversion process..."
python3 convert_to_parquet.py "$input_path" "$output_dir"

result=$?
if [ $result -eq 0 ]; then
    echo "Conversion completed successfully."
    echo "Parquet files are available in: $output_dir"
else
    echo "Conversion failed with exit code $result"
    exit $result
fi

# Optional: Clean up the temporary Python script
rm convert_to_parquet.py

# Optional: Display size comparison
echo ""
echo "Size comparison:"
if [ -d "$input_path" ]; then
    echo "Original data size: $(du -sh "$input_path" | cut -f1)"
else
    echo "Original file size: $(du -sh "$input_path" | cut -f1)"
fi
echo "Parquet data size: $(du -sh "$output_dir" | cut -f1)"
