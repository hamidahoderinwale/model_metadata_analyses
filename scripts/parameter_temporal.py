import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any, Optional
import re
import logging

# Import extraction utilities
from .metadata_extraction_utils import extract_parameters_from_metadata, extract_date_from_metadata, extract_model_family, extract_architecture_type

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_prepare_data(input_file: str) -> pd.DataFrame:
    """
    Load and prepare the data for parameter temporal analysis.
    Assumes the input CSV contains 'model_id' and a metadata column
    which could be named 'metadata' or 'card'.
    """
    logger.info(f"Loading data from {input_file}")
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Successfully loaded {len(df)} rows.")
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {input_file}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the CSV file: {e}")
        raise

    # Log initial data shape and column names
    logger.info(f"Initial data shape: {df.shape}")
    logger.info(f"Available columns: {df.columns.tolist()}")

    # Identify the metadata column (could be 'metadata' or 'card')
    metadata_col = None
    if 'metadata' in df.columns:
        metadata_col = 'metadata'
        logger.info("Using 'metadata' column for extraction.")
    elif 'card' in df.columns:
        metadata_col = 'card'
        logger.info("Using 'card' column for extraction.")
    else:
        logger.error("Required metadata column ('metadata' or 'card') not found.")
        # Create empty columns to prevent errors later
        df['metadata'] = np.nan
        df['card'] = np.nan
        # Continue processing, subsequent steps will likely filter out rows without metadata
        # Or handle NaNs gracefully

    # Ensure 'model_id' column exists
    if 'model_id' not in df.columns:
         logger.error("Required column 'model_id' not found.")
         # Create empty column to prevent errors
         df['model_id'] = "unknown_id" # Or just let it fail if essential


    # Apply extraction functions
    if metadata_col:
        logger.info("Extracting parameters, dates, family, and architecture from metadata.")
        df['parameters'] = df[metadata_col].apply(extract_parameters_from_metadata)
        df['created_at'] = df[metadata_col].apply(lambda x: extract_date_from_metadata(x, 'created_at'))
        df['last_modified'] = df[metadata_col].apply(lambda x: extract_date_from_metadata(x, 'last_modified'))
        family_info = df.apply(lambda row: extract_model_family(row['model_id'], row[metadata_col]), axis=1)
        df['model_family'] = family_info.apply(lambda x: x[0])
        df['family_confidence'] = family_info.apply(lambda x: x[1])
        df['architecture_type'] = df[metadata_col].apply(extract_architecture_type)
    else:
        # If no metadata column, fill extracted columns with NaNs/Unknowns
        logger.warning("No metadata column found. Extracted columns will be empty.")
        df['parameters'] = np.nan
        df['created_at'] = pd.NaT # Not a Time for datetime columns
        df['last_modified'] = pd.NaT
        df['model_family'] = "Unknown"
        df['family_confidence'] = "low"
        df['architecture_type'] = "Unknown"


    # Filter out models without valid parameters or creation date
    initial_count = len(df)
    df = df.dropna(subset=['parameters', 'created_at']).copy() # Use copy() to avoid SettingWithCopyWarning
    df = df[df['parameters'] > 0].copy() # Ensure parameters are positive
    df = df[df['created_at'].notna()].copy() # Explicitly check notna for datetime

    logger.info(f"Filtered down to {len(df)} models with valid parameters and creation dates out of {initial_count}.")

    # Add parameter size category
    # Define bins including 0 to handle edge cases correctly
    bins = [0, 1e6, 1e7, 1e8, 1e9, 1e10, float('inf')]
    labels = ['<1M', '1M-10M', '10M-100M', '100M-1B', '1B-10B', '>10B']
    df['parameter_category'] = pd.cut(df['parameters'], bins=bins, labels=labels, right=True, include_lowest=True)


    logger.info(f"Prepared data shape: {df.shape}")
    return df

def analyze_parameter_trends(df: pd.DataFrame, output_dir: str):
    """
    Analyze temporal trends in model parameters.
    Generates plots and summary statistics.
    """
    logger.info("\nAnalyzing parameter trends over time...")

    # Check if DataFrame is empty before proceeding
    if df.empty:
        logger.warning("DataFrame is empty after preparation. Skipping analysis and plotting.")
        # Generate empty stats and return
        empty_stats = {
            'total_models_in_input': pd.read_csv(os.path.join('/Users/hamidahoderinwale/Downloads/joined_models_20250529_135015.csv')).shape[0], # Re-read or pass original count if available
            'models_analyzed': 0,
            'date_range': None,
            'parameter_stats': {
                'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0
            },
            'family_stats': {},
            'architecture_stats': {},
            'models_per_year': {},
            'monthly_averages': {}
        }
        # Save empty statistics file to indicate analysis ran but found no data
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"parameter_temporal_analysis_{timestamp}")
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, 'parameter_temporal_summary.json'), 'w') as f:
                json.dump(empty_stats, f, indent=4)
            logger.info(f"Saved empty statistics to {os.path.join(output_path, 'parameter_temporal_summary.json')}")
        except Exception as e:
            logger.error(f"Error saving empty stats file: {e}")
        return empty_stats


    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"parameter_temporal_analysis_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Saving results to: {output_path}")

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid') # Using a more modern seaborn style

    # 1. Enhanced Parameter Distribution Plot
    logger.info("Generating Parameter Distribution vs Creation Date plot...")
    plt.figure(figsize=(15, 8))

    # Create scatter plot with model families as colors and architecture types as markers
    # Limit the number of families and architecture types for clarity in plotting and legend
    families_to_plot = df['model_family'].value_counts().nlargest(10).index.tolist()
    arch_types_to_plot = df['architecture_type'].value_counts().nlargest(5).index.tolist() # Limit architecture types
    arch_marker_map = {arch: marker for arch, marker in zip(arch_types_to_plot, ['o', 's', 'X', 'D', 'P'])} # Assign specific markers

    # Use 'Other Family' and 'Other Architecture' for less frequent categories
    df['family_for_plot'] = df['model_family'].apply(lambda x: x if x in families_to_plot else 'Other Family')
    df['arch_for_plot'] = df['architecture_type'].apply(lambda x: x if x in arch_types_to_plot else 'Other Architecture')


    # Generate a color palette large enough for all families + 'Other Family'
    all_families_in_plot = df['family_for_plot'].unique()
    colors = sns.color_palette('husl', n_colors=len(all_families_in_plot))
    family_colors = dict(zip(all_families_in_plot, colors))

    # Plot each family with different markers for architecture types within that family
    for family in all_families_in_plot:
        family_data = df[df['family_for_plot'] == family]
        for arch_type in family_data['arch_for_plot'].unique():
            arch_data = family_data[family_data['arch_for_plot'] == arch_type]
            # Get marker for the architecture type, default to 'o' if 'Other Architecture' or not in map
            marker = arch_marker_map.get(arch_type, 'o')

            # Ensure data for scatter plot is not empty
            if not arch_data.empty:
                plt.scatter(arch_data['created_at'], arch_data['parameters'],
                           c=[family_colors[family]], marker=marker, alpha=0.6,
                            label=f'{family} ({arch_type})')

    plt.yscale('log')
    plt.title('Model Parameters vs Creation Date by Family and Architecture (Top Categories)')
    plt.xlabel('Creation Date')
    plt.ylabel('Number of Parameters (log scale)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Enhanced grid
    plt.xticks(rotation=45, ha='right')

    # Add trend line - use all data with valid parameters and dates
    # Convert dates to numerical representation (seconds since epoch or a fixed point)
    # Using total_seconds() from timedelta for numerical x-axis for polyfit
    try:
        df_trend = df.dropna(subset=['created_at', 'parameters']).copy()
        if not df_trend.empty:
            # Use seconds since the earliest date in the dataset
            time_numeric = (df_trend['created_at'] - df_trend['created_at'].min()).dt.total_seconds()
            # Fit polynomial in log scale of parameters
            z = np.polyfit(time_numeric, np.log10(df_trend['parameters']), 1)
            p = np.poly1d(z)
            # Predict log parameters using the fitted polynomial and convert back to original scale
            # Use the full range of created_at for plotting the trend line
            date_range_numeric = (df['created_at'] - df['created_at'].min()).dt.total_seconds()
            # Ensure unique and sorted dates for plotting the trend line smoothly
            unique_dates_numeric = np.sort(date_range_numeric.unique())
            predicted_params = 10**p(unique_dates_numeric)

            # Convert unique_dates_numeric back to datetime for plotting
            unique_dates = df['created_at'].min() + pd.to_timedelta(unique_dates_numeric, unit='s')

            plt.plot(unique_dates, predicted_params, 'k--',
                     label=f'Trend ($10^{{{z[0]:.2e} \\times \\text{{time}} + {z[1]:.2f}}})$') # Latex formatted label
        else:
            logger.warning("Not enough data points with parameters and dates to plot trend line.")
    except Exception as e:
        logger.error(f"Error generating trend line: {e}")


    # Add legend with a reasonable number of entries
    handles, labels = plt.gca().get_legend_handles_labels()
    # Filter out labels that might not have corresponding handles if some data is missing
    valid_handles_labels = [(h, l) for h, l in zip(handles, labels) if h is not None]
    by_label = dict(valid_handles_labels)

    # Adjust legend location and size if too many items
    if len(by_label) > 20: # Arbitrary threshold for complexity
         plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    else:
         plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig(os.path.join(output_path, 'parameters_vs_creation_enhanced.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Parameter Distribution plot saved.")

    # 2. Parameter Size Distribution by Family (Top Families)
    logger.info("Generating Parameter Distribution by Family plot...")
    if not df['family_for_plot'].nunique() > 1:
         logger.warning("Not enough unique model families to generate box plot by family. Skipping.")
    else:
        plt.figure(figsize=(15, 8))
        # Use the family_for_plot column which groups less frequent families
        sns.boxplot(data=df, x='family_for_plot', y='parameters', order=df['family_for_plot'].value_counts().index) # Order by frequency
        plt.yscale('log')
        plt.title('Parameter Size Distribution by Model Family (Top Categories)')
        plt.xlabel('Model Family')
        plt.ylabel('Number of Parameters (log scale)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Enhanced grid
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'parameter_distribution_by_family.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Parameter Distribution by Family plot saved.")

    # 3. Monthly Average Parameter Size
    logger.info("Generating Monthly Average Parameter Size plot...")
    # Group by month period and calculate mean parameters
    monthly_avg = df.groupby(df['created_at'].dt.to_period('M'))['parameters'].mean()
    # Convert period index to string for plotting
    monthly_avg.index = monthly_avg.index.astype(str)

    if monthly_avg.empty or len(monthly_avg) < 2:
         logger.warning("Not enough data points across months to generate monthly average trend plot. Skipping.")
    else:
        plt.figure(figsize=(15, 8))
        plt.plot(monthly_avg.index, monthly_avg.values, 'b-', marker='o', linestyle='-')
        plt.yscale('log')
        plt.title('Monthly Average Model Parameters (Models with Known Parameters and Dates)')
        plt.xlabel('Month')
        plt.ylabel('Average Number of Parameters (log scale)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Enhanced grid
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'monthly_parameter_trend.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Monthly Average Parameter Size plot saved.")


    # 4. Parameter Size Distribution Over Time (by Year)
    logger.info("Generating Parameter Size Distribution by Year plot...")
    # Extract year for grouping
    df['year'] = df['created_at'].dt.year
    if not df['year'].nunique() > 1:
         logger.warning("Not enough unique years to generate box plot by year. Skipping.")
    else:
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=df, x='year', y='parameters', order=sorted(df['year'].unique())) # Order by year
        plt.yscale('log')
        plt.title('Parameter Size Distribution by Year (Models with Known Parameters and Dates)')
        plt.xlabel('Year')
        plt.ylabel('Number of Parameters (log scale)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Enhanced grid
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'yearly_parameter_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Parameter Size Distribution by Year plot saved.")


    # Generate enhanced summary statistics
    logger.info("Generating summary statistics...")
    # Recalculate family stats using the original model_family column for full detail
    family_stats_detail = df.groupby('model_family').agg({
        'parameters': ['count', 'mean', 'median', 'min', 'max']
    }).round(3)
    # Convert MultiIndex columns to single level for JSON serialization
    family_stats_detail.columns = ['_'.join(col).strip() for col in family_stats_detail.columns.values]

    architecture_stats_detail = df.groupby('architecture_type').agg({
         'parameters': ['count', 'mean', 'median', 'min', 'max']
    }).round(3)
    architecture_stats_detail.columns = ['_'.join(col).strip() for col in architecture_stats_detail.columns.values]

    summary_stats = {
        # Use the total number of models loaded from the input file before filtering
        'total_models_in_input': pd.read_csv(os.path.join('/Users/hamidahoderinwale/Downloads/joined_models_20250529_135015.csv')).shape[0], # Re-read or pass original count if available
        'models_analyzed': len(df),
        'date_range': {
            'creation': {
                'start': df['created_at'].min().strftime('%Y-%m-%d %H:%M:%S') if not df.empty else None,
                'end': df['created_at'].max().strftime('%Y-%m-%d %H:%M:%S') if not df.empty else None
            },
            # Include last_modified date range if data exists
            'modification': {
                'start': df['last_modified'].min().strftime('%Y-%m-%d %H:%M:%S') if df['last_modified'].notna().any() else None,
                'end': df['last_modified'].max().strftime('%Y-%m-%d %H:%M:%S') if df['last_modified'].notna().any() else None
            }
        },
        'parameter_stats': {
            'mean': float(df['parameters'].mean()) if not df.empty else 0.0,
            'median': float(df['parameters'].median()) if not df.empty else 0.0,
            'min': float(df['parameters'].min()) if not df.empty else 0.0,
            'max': float(df['parameters'].max()) if not df.empty else 0.0,
            'std': float(df['parameters'].std()) if not df.empty else 0.0,
        },
        'family_stats': family_stats_detail.to_dict('index'), # Save as dictionary oriented by index (family)
        'architecture_stats': architecture_stats_detail.to_dict('index'), # Save as dictionary oriented by index (architecture)
        'models_per_year': df['year'].value_counts().sort_index().to_dict() if not df.empty else {}, # Models analyzed per year
        'monthly_averages': {str(period): float(avg) for period, avg in monthly_avg.items()} if not monthly_avg.empty else {}
    }

    # Save statistics
    try:
        with open(os.path.join(output_path, 'parameter_temporal_summary.json'), 'w') as f:
            json.dump(summary_stats, f, indent=4)
        logger.info("Summary statistics saved.")
    except Exception as e:
        logger.error(f"Error saving summary statistics: {e}")

    logger.info("Parameter temporal analysis complete.")
    return summary_stats

def main():
    # Determine base output directory relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir, 'outputs')
    output_dir = os.path.join(base_output_dir, 'parameter_temporal')

    # Ensure the base output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Base output directory: {output_dir}")

    # Use the specific CSV file path
    # Consider making this an argument later
    input_path = '/Users/hamidahoderinwale/Downloads/joined_models_20250529_135015.csv'
    logger.info(f"Using data from: {input_path}")

    try:
        df = load_and_prepare_data(input_path)
        stats = analyze_parameter_trends(df, output_dir)

        # Print summary of findings (using stats dictionary)
        print("\nParameter Temporal Analysis Summary:")
        print(f"Total models loaded from input: {stats.get('total_models_in_input', 'N/A')}")
        print(f"Models analyzed with valid parameters and dates: {stats.get('models_analyzed', 0)}")

        if stats.get('date_range') and stats['date_range'].get('creation'):
            creation_range = stats['date_range']['creation']
            print(f"Creation date range: {creation_range.get('start', 'N/A')} to {creation_range.get('end', 'N/A')}")

        if stats.get('parameter_stats') and stats['models_analyzed'] > 0:
            param_stats = stats['parameter_stats']
            print("\nParameter Statistics (Analyzed Models):")
            print(f"Mean parameters: {param_stats.get('mean', 0.0):,.2f}") # Format with comma
            print(f"Median parameters: {param_stats.get('median', 0.0):,.2f}")
            print(f"Min parameters: {param_stats.get('min', 0.0):,.2f}")
            print(f"Max parameters: {param_stats.get('max', 0.0):,.2f}")

        if stats.get('models_per_year'):
            print("\nModels analyzed per year:")
            # Sort years for consistent output
            sorted_years = sorted(stats['models_per_year'].keys())
            for year in sorted_years:
                print(f"{year}: {stats['models_per_year'][year]} models")

        print(f"Detailed results (plots and stats JSON) saved in a timestamped folder inside: {output_dir}")

    except FileNotFoundError:
        logger.error("Analysis aborted due to file not found error.")
    except Exception as e:
        logger.error(f"Analysis aborted due to an error: {e}", exc_info=True) # Log traceback

if __name__ == "__main__":
    main() 
