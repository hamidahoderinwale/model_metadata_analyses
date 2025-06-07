import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import yaml
from scipy import stats
from typing import Tuple, List, Dict, Any, Optional
import logging
from collections import defaultdict
import re
import ast # Added for literal_eval if needed from parameter_scatter

# Import extraction utilities - keeping extract_date_from_metadata and adding extract_parameters_from_metadata
from metadata_extraction_utils import extract_date_from_metadata, extract_parameters_from_metadata

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Start of functions copied from parameter_scatter.py ---
def parse_number(text):
    """Parse number strings with K/M/B suffixes into float values."""
    if pd.isna(text):
        return np.nan
    try:
        text = str(text).strip().replace(',', '').lower()
        if 'e' in text:
            return float(text)
        multipliers = {'k': 1e3, 'm': 1e6, 'b': 1e9}
        for suffix, mult in multipliers.items():
            if text.endswith(suffix):
                return float(text[:-1]) * mult
        return float(text)
    except Exception:
        return np.nan

def extract_parameters_from_card(card_text):
    """Extract parameter counts from unstructured card text using multiple regex patterns."""
    if pd.isna(card_text):
        return None

    param_patterns = [
        r'parameters:\s*{\s*"F32":\s*(\d+)\s*}',  # parameters: {"F32": 85804039}
        r'"parameters":\s*{\s*"F32":\s*(\d+)\s*}', # "parameters": {"F32": 85804039}
        r'parameters:\s*(\d+(?:\.\d+)?[BKM]?)',  # parameters: 127k
        r'Total parameters:\s*(\d+(?:\.\d+)?[BKM]?)', # Total parameters: 127k
        r'~?(\d+(?:\.\d+)?[BKM]?)\s*parameters',  # ~127k parameters
        r'Params:\s*(\d+(?:\.\d+)?[BKM]?)',  # Params: 127k
        r'#Params:\s*(\d+(?:\.\d+)?[BKM]?)', # #Params: 127k
        r'(\d+(?:\.\d+)?[BKM]?)\s*params',   # 127k params
        r'(\d+(?:\.\d+)?[BKM]?)\s*parameters', # 127k parameters
        r'"num_parameters":\s*(\d+(?:\.\d+)?[BKM]?)', # "num_parameters": 127k
        r'"total_parameters":\s*(\d+(?:\.\d+)?[BKM]?)', # "total_parameters": 127k
    ]
    for pattern in param_patterns:
        match = re.search(pattern, card_text, re.IGNORECASE)
        if match:
            return parse_number(match.group(1))
    return None

def extract_metadata_fields(metadata_str):
    """Extract parameter count from metadata string (robust to different formats)."""
    if pd.isna(metadata_str):
        return pd.Series({'parameters': None})

    metadata = None
    if isinstance(metadata_str, str):
        try:
            # First try ast.literal_eval as it's more flexible for Python literals
            metadata = ast.literal_eval(metadata_str)
        except (ValueError, SyntaxError):
            try:
                # Fallback to json.loads, cleaning single quotes
                cleaned_str = metadata_str.replace("'", '"').strip()
                metadata = json.loads(cleaned_str)
            except json.JSONDecodeError:
                # If all parsing fails, try to extract parameters directly using regex
                param_match = re.search(r'"parameters":\s*(\d+)', cleaned_str)
                if param_match:
                    return pd.Series({'parameters': float(param_match.group(1))})
                return pd.Series({'parameters': None})
    else:
        # If input is already a dict, use it directly
        metadata = metadata_str

    # Extract parameters from the parsed metadata (if it's a dictionary)
    if isinstance(metadata, dict):
        # Try common parameter keys
        param_keys = ['parameters', 'num_parameters', 'total_parameters', 'n_parameters', 'size', 'model_size']
        for key in param_keys:
            if key in metadata:
                value = metadata[key]
                if isinstance(value, (int, float)):
                    return pd.Series({'parameters': float(value)})
                elif isinstance(value, str):
                    parsed = parse_number(value)
                    if not np.isnan(parsed):
                        return pd.Series({'parameters': parsed})
                elif isinstance(value, dict):
                    # Handle nested parameter dictionaries like {"F32": 85804039}
                    for v in value.values():
                        parsed = parse_number(v)
                        if not np.isnan(parsed):
                            return pd.Series({'parameters': parsed})
        
        # Add specific nested checks (from parameter_distribution.py)
        if 'model' in metadata and isinstance(metadata['model'], dict) and 'parameters' in metadata['model']:
            value = metadata['model']['parameters']
            parsed = parse_number(value)
            if not np.isnan(parsed):
                return pd.Series({'parameters': parsed})
        
        if 'config' in metadata and isinstance(metadata['config'], dict) and 'parameters' in metadata['config']:
            value = metadata['config']['parameters']
            parsed = parse_number(value)
            if not np.isnan(parsed):
                return pd.Series({'parameters': parsed})

        # Search nested dictionaries recursively
        for v in metadata.values():
            if isinstance(v, dict):
                # Pass the dictionary directly to the function
                result = extract_metadata_fields(v) 
                if result['parameters'] is not None:
                    return result

    return pd.Series({'parameters': None})
# --- End of functions copied from parameter_scatter.py ---

def load_and_prepare_data(input_file: str) -> pd.DataFrame:
    """Load and prepare the data for analysis."""
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    logger.info(f"Initial data shape: {df.shape}")
    logger.info(f"Available columns: {df.columns.tolist()}")
    
    df['parameters'] = np.nan
    df['created_at'] = pd.NaT
    df['last_modified'] = pd.NaT

    logger.info("Attempting to extract parameters and dates from multiple potential columns.")
    
    # --- Parameter Extraction Logic ---
    # Create temporary columns for extracted parameters from different sources
    df['temp_params_metadata'] = np.nan
    df['temp_params_card'] = np.nan
    df['temp_params_model_id'] = np.nan

    # 1. Extract from 'metadata' column (most structured)
    if 'metadata' in df.columns:
        logger.debug("Applying extract_metadata_fields to 'metadata' column.")
        df['temp_params_metadata'] = df['metadata'].apply(lambda x: extract_metadata_fields(x)['parameters'])
        
    # 2. Extract from 'card' column (model card text)
    if 'card' in df.columns:
        logger.debug("Applying extract_parameters_from_card to 'card' column.")
        df['temp_params_card'] = df['card'].apply(extract_parameters_from_card)

    # 3. Extract from 'model_id' column (e.g., "llama-7b", "gemma-2b")
    # Using the more general `extract_parameters_from_metadata` from `metadata_extraction_utils`
    # for the model_id column, as it handles plain strings with suffixes like '7b' or '2b'.
    if 'model_id' in df.columns:
        logger.debug("Applying metadata_extraction_utils.extract_parameters_from_metadata to 'model_id' column.")
        df['temp_params_model_id'] = df['model_id'].apply(extract_parameters_from_metadata)

    # Consolidate parameters, prioritizing in order: metadata > card > model_id
    # Initialize with metadata params
    df['parameters'] = df['temp_params_metadata']
    # Fill missing from card params
    df['parameters'] = df['parameters'].fillna(df['temp_params_card'])
    # Fill remaining missing from model_id params
    df['parameters'] = df['parameters'].fillna(df['temp_params_model_id'])

    # Track the ultimate source of the parameter
    df['parameter_source'] = 'unknown'
    df.loc[df['temp_params_metadata'].notna(), 'parameter_source'] = 'metadata'
    df.loc[df['temp_params_metadata'].isna() & df['temp_params_card'].notna(), 'parameter_source'] = 'card'
    df.loc[df['temp_params_metadata'].isna() & df['temp_params_card'].isna() & df['temp_params_model_id'].notna(), 'parameter_source'] = 'model_id'

    # Clean up temporary source columns
    df = df.drop(columns=['temp_params_metadata', 'temp_params_card', 'temp_params_model_id'], errors='ignore')

    # --- Date Extraction Logic ---
    potential_date_cols = ['metadata', 'card', 'model_id']
    for index, row in df.iterrows():
        extracted_created_at_val = pd.NaT
        extracted_last_modified_val = pd.NaT
        
        for col in potential_date_cols:
            if col in df.columns and pd.notna(row[col]):
                current_content = str(row[col])

                if pd.isna(extracted_created_at_val):
                    extracted_created_at = extract_date_from_metadata(current_content, 'created_at')
                    if extracted_created_at is not None:
                        extracted_created_at_val = extracted_created_at

                if pd.isna(extracted_last_modified_val):
                    extracted_last_modified = extract_date_from_metadata(current_content, 'last_modified')
                    if extracted_last_modified is not None:
                        extracted_last_modified_val = extracted_last_modified
            
            if not pd.isna(extracted_created_at_val) and not pd.isna(extracted_last_modified_val):
                break
        
        df.at[index, 'created_at'] = extracted_created_at_val
        df.at[index, 'last_modified'] = extracted_last_modified_val

    # Log parameter and date extraction counts
    logger.info("Parameter extraction summary by source:")
    if 'parameter_source' in df.columns:
        for source, count in df['parameter_source'].value_counts().items():
            logger.info(f"  From {source}: {count} models")
    
    logger.info("Date extraction summary:")
    logger.info(f"  Created At: {df['created_at'].count()} models")
    logger.info(f"  Last Modified: {df['last_modified'].count()} models")

    # Filter out models without valid parameters or creation date
    initial_count = len(df)
    df_filtered = df.dropna(subset=['parameters', 'created_at']).copy()
    df_filtered = df_filtered[df_filtered['parameters'] > 0].copy()

    if not df_filtered.empty and pd.api.types.is_numeric_dtype(df_filtered['parameters']):
        if df_filtered['parameters'].dropna().apply(lambda x: np.isclose(x, round(x))).all():
            df_filtered['parameters'] = df_filtered['parameters'].astype(int)
        else:
            logger.warning("Parameters column contains non-integer float values. Mathematical property analysis might be less accurate.")

    logger.info(f"Filtered down to {len(df_filtered)} models with valid parameters and creation dates out of {initial_count}.")

    valid_params_count = df_filtered['parameters'].dropna().count()
    valid_created_at_count = df_filtered['created_at'].dropna().count()
    valid_last_modified_count = df_filtered['last_modified'].dropna().count()

    logger.info(f"Models with valid parameters: {valid_params_count}")
    logger.info(f"Models with valid creation dates: {valid_created_at_count}")
    logger.info(f"Models with valid last modified dates: {valid_last_modified_count}")

    bins = [0, 1e6, 1e7, 1e8, 1e9, 1e10, float('inf')]
    labels = ['<1M', '1M-10M', '10M-100M', '100M-1B', '1B-10B', '>10B']
    if not df_filtered.empty:
        df_filtered['parameter_category'] = pd.cut(df_filtered['parameters'], bins=bins, labels=labels, right=True, include_lowest=True)
    else:
        df_filtered['parameter_category'] = pd.Series(dtype='category')

    logger.info(f"Prepared data shape: {df_filtered.shape}")
    return df_filtered

def analyze_mathematical_properties(parameters: np.ndarray) -> Dict:
    """Analyze mathematical properties of parameter counts."""
    properties = {
        'powers_of_2': [],
        'powers_of_10': [],
        'round_numbers': [],
        'prime_numbers': [],
        'highly_composite': []
    }
    
    def is_power_of_2(n):
        # Check if n is a positive integer or a positive float representing an integer
        if not isinstance(n, (int, np.integer, float, np.floating)) or n <= 0 or not np.isfinite(n):
            return False
        # Check if it's effectively an integer
        if not np.isclose(n, round(n)):
            return False
        # Convert to integer for bitwise operation
        int_n = int(round(n))
        return (int_n > 0) and ((int_n & (int_n - 1)) == 0)
    
    def is_power_of_10(n):
        # Check if n is a positive number and effectively an integer
        if not isinstance(n, (int, np.integer, float, np.floating)) or n <= 0 or not np.isfinite(n):
            return False
        if not np.isclose(n, round(n)):
            return False
        int_n = int(round(n))
        # Check if it's a power of 10 (e.g., 1, 10, 100, 1000)
        # Use log10 and check if the result is a non-negative integer
        try:
            log10_n = np.log10(int_n)
            return np.isclose(log10_n, round(log10_n), atol=1e-9) and round(log10_n) >= 0
        except (ValueError, RuntimeWarning):
            return False
    
    def is_round_number(n):
        # Check if n is a positive number and effectively an integer
        if not isinstance(n, (int, np.integer, float, np.floating)) or n <= 0 or not np.isfinite(n):
            return False
        if not np.isclose(n, round(n)):
            return False
        
        # Convert to integer for string manipulation
        int_n = int(round(n))
        
        # Check if number is "round" (ends in 0 or 5 after removing trailing zeros, or is small)
        s = str(int_n).rstrip('0')
        return s == '' or s[-1] in '05' or len(s) <= 2
    
    def is_prime(n):
        if not isinstance(n, (int, np.integer, float, np.floating)) or n < 2 or not np.isfinite(n) or not np.isclose(n, round(n)):
            return False
        n_int = int(round(n))
        for i in range(2, int(np.sqrt(n_int)) + 1):
            if n_int % i == 0:
                return False
        return True
    
    def is_highly_composite(n):
        # Count number of divisors
        if not isinstance(n, (int, np.integer, float, np.floating)) or n < 1 or not np.isfinite(n) or not np.isclose(n, round(n)):
            return False
        n_int = int(round(n))
        divisors = 0
        for i in range(1, int(np.sqrt(n_int)) + 1):
            if n_int % i == 0:
                divisors += 2 if i * i != n_int else 1
        return divisors > 12
    
    for param in parameters:
        if pd.isna(param) or param <= 0:
            continue # Skip NaN or non-positive values

        # Check for powers of 2 and 10 (can be float)
        if is_power_of_2(param):
             properties['powers_of_2'].append(float(param))
        if is_power_of_10(param):
             properties['powers_of_10'].append(float(param))
        if is_round_number(param):
             properties['round_numbers'].append(float(param))

    # Check for prime and highly composite (must be positive integers)
    # The explicit check in the functions themselves makes this filtering less critical, but good for clarity
    # Simplified: now just iterate over all valid parameters, as functions handle type checking internally
    for param in parameters[np.isfinite(parameters) & (parameters > 0)]:
        if is_prime(param):
            properties['prime_numbers'].append(float(param))
        if is_highly_composite(param):
            properties['highly_composite'].append(float(param))


    # Ensure unique values in properties lists if needed, though not strictly necessary for counts/histograms
    for key in properties:
        properties[key] = sorted(list(set(properties[key])))
    
    return properties

def find_mathematical_gaps(parameters: np.ndarray) -> Dict:
    """Find gaps in parameter choices based on mathematical properties."""
    # Sort parameters
    sorted_params = np.sort(parameters)
    
    # Calculate gaps between consecutive parameters
    gaps = np.diff(sorted_params)
    
    # Find significant gaps (e.g., gaps larger than 2 standard deviations)
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)
    significant_gaps = gaps > (mean_gap + 2 * std_gap)
    
    # Get the parameters before and after each significant gap
    gap_info = []
    for i in np.where(significant_gaps)[0]:
        gap_info.append({
            'before': float(sorted_params[i]),
            'after': float(sorted_params[i + 1]),
            'gap_size': float(gaps[i]),
            'relative_gap': float(gaps[i] / mean_gap)
        })
    
    return {
        'gap_info': gap_info,
        'mean_gap': float(mean_gap),
        'std_gap': float(std_gap)
    }

def analyze_sloane_patterns(df: pd.DataFrame, output_dir: str):
    """Analyze parameter patterns similar to Sloane's integer sequence analysis."""
    if len(df) == 0:
        logger.error("No valid data to analyze")
        return None
        
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"sloane_gap_analysis_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the processed DataFrame to a CSV file for model-level attribution
    analyzed_models_csv_path = output_path / 'analyzed_models_data.csv'
    df.to_csv(analyzed_models_csv_path, index=False)
    logger.info(f"Saved processed model data with extracted parameters and dates to {analyzed_models_csv_path}")
    
    # Set style
    plt.style.use('default')
    
    # 1. Analyze mathematical properties
    logger.info("Analyzing mathematical properties of parameters.")
    properties = analyze_mathematical_properties(df['parameters'].values)
    
    # Plot distribution of mathematical properties
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each property type
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    property_types = ['powers_of_2', 'powers_of_10', 'round_numbers', 
                     'prime_numbers', 'highly_composite']
    titles = ['Powers of 2', 'Powers of 10', 'Round Numbers', 
             'Prime Numbers', 'Highly Composite Numbers']
    
    for idx, (prop_type, title) in enumerate(zip(property_types, titles)):
        if idx < len(axes):
            values = properties[prop_type]
            if values:
                sns.histplot(values, ax=axes[idx], bins=20)
                axes[idx].set_title(f'Distribution of {title}')
                axes[idx].set_xlabel('Parameter Count')
                axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_path / 'mathematical_properties.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Analyze gaps
    gap_analysis = find_mathematical_gaps(df['parameters'].values)
    
    # Plot parameter distribution with gaps
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(df)), np.sort(df['parameters']), alpha=0.5, s=10)
    
    # Mark significant gaps
    for gap in gap_analysis['gap_info']:
        plt.axvline(x=np.where(np.sort(df['parameters']) == gap['before'])[0][0], 
                   color='r', alpha=0.3, linestyle='--')
    
    plt.yscale('log')
    plt.title('Parameter Distribution with Mathematical Gaps')
    plt.xlabel('Model Index (sorted by parameters)')
    plt.ylabel('Number of Parameters (log scale)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'mathematical_gaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Analyze parameter clustering
    # Use KDE to identify clusters
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df['parameters'], log_scale=True)
    plt.title('Parameter Density with Mathematical Clusters')
    plt.xlabel('Number of Parameters (log scale)')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'parameter_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics
    stats = {
        'mathematical_properties': {
            prop_type: {
                'count': len(values),
                'percentage': len(values) / len(df) * 100 if len(df) > 0 else 0,
                'values': sorted(values)
            }
            for prop_type, values in properties.items()
        },
        'gap_analysis': {
            'total_gaps': len(gap_analysis['gap_info']),
            'mean_gap': float(gap_analysis['mean_gap']),
            'std_gap': float(gap_analysis['std_gap']),
            'significant_gaps': gap_analysis['gap_info']
        },
        'overall_stats': {
            'total_models': len(df),
            'mean_parameters': float(df['parameters'].mean()) if len(df) > 0 else 0,
            'median_parameters': float(df['parameters'].median()) if len(df) > 0 else 0,
            'std_parameters': float(df['parameters'].std()) if len(df) > 0 else 0,
            'min_parameters': float(df['parameters'].min()) if len(df) > 0 else 0,
            'max_parameters': float(df['parameters'].max()) if len(df) > 0 else 0
        }
    }
    
    with open(output_path / 'sloane_analysis_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Analysis complete. Results saved to {output_path}")
    return stats

def main():
    input_file = "/Users/hamidahoderinwale/Downloads/joined_models_20250529_135015.csv"
    output_dir = "outputs/parameter_analysis"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        df = load_and_prepare_data(input_file)
        if len(df) == 0:
            logger.error("No valid models found with parameter counts")
            return
            
        stats = analyze_sloane_patterns(df, output_dir)
        if stats is None:
            return
        
        # Print summary of findings
        print("\nSloane Gap Analysis Summary:")
        print(f"Total models analyzed: {stats['overall_stats']['total_models']}")
        
        print("\nMathematical Properties:")
        for prop_type, prop_stats in stats['mathematical_properties'].items():
            print(f"\n{prop_type.replace('_', ' ').title()}:")
            print(f"Count: {prop_stats['count']} ({prop_stats['percentage']:.1f}%)")
            if prop_stats['values']:
                print(f"Examples: {prop_stats['values'][:5]}")
        
        print("\nSignificant Gaps:")
        for gap in stats['gap_analysis']['significant_gaps'][:5]:  # Show top 5 gaps
            print(f"Gap between {gap['before']:.2e} and {gap['after']:.2e} "
                  f"(size: {gap['gap_size']:.2e}, {gap['relative_gap']:.1f}x mean)")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
