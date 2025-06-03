import os
import json
import argparse
from datetime import datetime
import re
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Methodology:
This script performs a bootstrap analysis of model parameter distributions to identify and analyze gaps in model sizes.
It first extracts parameter counts from both model metadata (preferred) and model IDs (fallback), then uses bootstrap
resampling to calculate confidence intervals for key statistics. The analysis focuses on identifying the parameter gap
between 20B and 70B parameters, while also providing distribution analysis across different parameter ranges.
"""

class BootstrapAnalyzer:
    """Performs bootstrap analysis on model parameter data."""

    def __init__(self, seed=42, n_bootstrap=1000):
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.n_bootstrap = n_bootstrap

        self.param_ranges = {
            'Tiny (0-1M)': (0, 1e6),
            'Small (1M-10M)': (1e6, 1e7),
            'Medium (10M-100M)': (1e7, 1e8),
            'Large (100M-1B)': (1e8, 1e9),
            'Huge (1B-20B)': (1e9, 20e9),
            'Gap (20B-70B)': (20e9, 70e9),
            'Massive (>70B)': (70e9, float('inf')),
        }

    def extract_parameters_from_metadata(self, metadata_str):
        """Extract parameter count from metadata JSON string."""
        if pd.isna(metadata_str) or not metadata_str.strip():
            return None
            
        try:
            # Parse the metadata string as JSON
            metadata = json.loads(metadata_str)
            
            # Look for safetensors parameters
            if 'safetensors' in metadata and 'parameters' in metadata['safetensors']:
                params = metadata['safetensors']['parameters']
                if 'total' in params:
                    return float(params['total'])
                elif isinstance(params, dict):
                    # Sum all parameter types (F32, F16, etc.)
                    total = sum(float(v) for v in params.values() if isinstance(v, (int, float)))
                    return total if total > 0 else None
                    
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            # If JSON parsing fails, try to extract from model_id as fallback
            pass
            
        return None

    def extract_parameters_from_model_id(self, model_id):
        """Extract parameter count from model_id as fallback method."""
        if pd.isna(model_id):
            return None
            
        # Look for patterns like "14B", "7B", etc.
        match = re.search(r'(\d+(?:\.\d+)?)[Bb]', str(model_id))
        if match:
            return float(match.group(1)) * 1e9
            
        # Look for patterns like "14M", "7M", etc.
        match = re.search(r'(\d+(?:\.\d+)?)[Mm]', str(model_id))
        if match:
            return float(match.group(1)) * 1e6
            
        return None

    def extract_parameters(self, row):
        """Extract parameter count from metadata first, then model_id as fallback."""
        # First try to get from metadata
        params = self.extract_parameters_from_metadata(row['metadata'])
        if params is not None:
            return params
            
        # Fallback to model_id extraction
        return self.extract_parameters_from_model_id(row['model_id'])

    def bootstrap_sample(self, data):
        """Return a bootstrap sample from the dataset."""
        return data.iloc[self.rng.randint(0, len(data), size=len(data))]

    def calculate_statistics(self, df):
        """Compute summary statistics and bin counts for a dataframe."""
        # Filter out rows where we couldn't extract parameters
        valid_df = df[df['parameters'].notna() & (df['parameters'] > 0)]
        
        if len(valid_df) == 0:
            # Return empty stats if no valid data
            return {
                'parameter_distribution': {
                    'mean': 0, 'median': 0, 'std': 0, 'q1': 0, 'q3': 0,
                },
                'depth_parameter_correlation': 0,
                'gap_analysis': {
                    'gap_percentage': 0,
                    'models_below_gap': 0,
                    'models_above_gap': 0,
                    'models_in_gap': 0,
                },
                'parameter_bins': {bin_name: 0 for bin_name in self.param_ranges.keys()},
            }
        
        return {
            'parameter_distribution': {
                'mean': valid_df['parameters'].mean(),
                'median': valid_df['parameters'].median(),
                'std': valid_df['parameters'].std(),
                'q1': valid_df['parameters'].quantile(0.25),
                'q3': valid_df['parameters'].quantile(0.75),
            },
            'depth_parameter_correlation': valid_df[['depth', 'parameters']].corr().iloc[0, 1] if 'depth' in valid_df.columns else 0,
            'gap_analysis': {
                'gap_percentage': len(valid_df[(valid_df['parameters'] >= 20e9) & (valid_df['parameters'] <= 70e9)]) / len(valid_df) * 100,
                'models_below_gap': len(valid_df[valid_df['parameters'] < 20e9]),
                'models_above_gap': len(valid_df[valid_df['parameters'] > 70e9]),
                'models_in_gap': len(valid_df[(valid_df['parameters'] >= 20e9) & (valid_df['parameters'] <= 70e9)]),
            },
            'parameter_bins': {
                bin_name: len(valid_df[(valid_df['parameters'] >= r[0]) & (valid_df['parameters'] < r[1])])
                for bin_name, r in self.param_ranges.items()
            },
        }

    def bootstrap_analysis(self, df):
        """Run bootstrap iterations and calculate confidence intervals."""
        print(f"Performing {self.n_bootstrap} bootstrap iterations...")
        stats_list = []

        for _ in tqdm(range(self.n_bootstrap)):
            sample = self.bootstrap_sample(df)
            stats_list.append(self.calculate_statistics(sample))

        def extract_ci(key, subkey=None):
            values = [s[key][subkey] if subkey else s[key] for s in stats_list]
            return {
                'mean': np.mean(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
            }

        result = {}
        for key in stats_list[0]:
            if isinstance(stats_list[0][key], dict):
                result[key] = {
                    subkey: extract_ci(key, subkey)
                    for subkey in stats_list[0][key]
                }
            else:
                result[key] = extract_ci(key)
        return result

    def plot_bootstrap_results(self, df, results, output_dir):
        """Visualize results and save plots."""
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('default')

        # Filter out rows where we couldn't extract parameters and ensure positive values
        valid_df = df[df['parameters'].notna() & (df['parameters'] > 0)]
        
        if len(valid_df) == 0:
            print("Warning: No valid parameter data found for plotting.")
            return

        # Plot 1: Parameter distribution histogram
        plt.figure(figsize=(15, 8))
        log_params = np.log10(valid_df['parameters'])
        plt.hist(log_params, bins=50, alpha=0.7, density=True, label='Original Data')
        plt.axvspan(np.log10(20e9), np.log10(70e9), color='red', alpha=0.2, label='Parameter Gap (20B-70B)')
        plt.xlabel('Log10(Parameters)')
        plt.ylabel('Density')
        plt.title('Parameter Distribution with Gap Region')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add x-axis labels for readability
        ticks = [6, 7, 8, 9, 10, 11, 12]  # 1M, 10M, 100M, 1B, 10B, 100B, 1T
        labels = ['1M', '10M', '100M', '1B', '10B', '100B', '1T']
        plt.xticks(ticks, labels)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_distribution.png'), dpi=300)
        plt.close()

        # Plot 2: Parameter bins
        bin_stats = results['parameter_bins']
        bins = list(bin_stats.keys())
        means = [bin_stats[b]['mean'] for b in bins]
        lowers = [bin_stats[b]['ci_lower'] for b in bins]
        uppers = [bin_stats[b]['ci_upper'] for b in bins]
        errs = [np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)]

        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(bins)), means, yerr=errs, capsize=5, alpha=0.7)
        plt.xticks(range(len(bins)), bins, rotation=45, ha='right')
        plt.ylabel('Model Count')
        plt.title('Models per Parameter Bin (with 95% CI)')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            if mean > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errs[1][i]/2, 
                        f'{int(mean)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_bins_distribution.png'), dpi=300)
        plt.close()

        # Plot 3: Gap analysis
        gap = results['gap_analysis']
        labels = ['Gap %', 'Below Gap', 'In Gap', 'Above Gap']
        keys = ['gap_percentage', 'models_below_gap', 'models_in_gap', 'models_above_gap']
        gap_means = [gap[k]['mean'] for k in keys]
        gap_errs = [
            [gap[k]['mean'] - gap[k]['ci_lower'] for k in keys],
            [gap[k]['ci_upper'] - gap[k]['mean'] for k in keys],
        ]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(keys)), gap_means, yerr=gap_errs, capsize=5, alpha=0.7)
        plt.xticks(range(len(keys)), labels)
        plt.ylabel('Count / Percentage')
        plt.title('Gap Analysis: 20B-70B Parameter Range (with 95% CI)')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, gap_means)):
            label = f'{mean:.1f}%' if i == 0 else f'{int(mean)}'
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + gap_errs[1][i]/2, 
                    label, ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gap_analysis.png'), dpi=300)
        plt.close()

        # Plot 4: Parameter extraction source breakdown
        metadata_params = df.apply(lambda row: self.extract_parameters_from_metadata(row['metadata']), axis=1).notna().sum()
        model_id_params = (df.apply(lambda row: self.extract_parameters_from_metadata(row['metadata']), axis=1).isna() & 
                          df.apply(lambda row: self.extract_parameters_from_model_id(row['model_id']), axis=1).notna()).sum()
        no_params = len(df) - metadata_params - model_id_params

        plt.figure(figsize=(10, 6))
        sources = ['From Metadata', 'From Model ID', 'No Parameters']
        counts = [metadata_params, model_id_params, no_params]
        colors = ['green', 'orange', 'red']
        
        bars = plt.bar(sources, counts, color=colors, alpha=0.7)
        plt.ylabel('Number of Models')
        plt.title('Parameter Extraction Source Breakdown')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                    f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_extraction_sources.png'), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Bootstrap analysis on model parameter data.')
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--input_file',
        type=str,
        default='/Users/hamidahoderinwale/Desktop/joined_models_20250529_135015.csv',
        help='Path to the input CSV file'
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"bootstrap_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    print(f"Loading data from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    # Extract parameters from both metadata and model_id
    print("Extracting parameters from metadata and model IDs...")
    analyzer = BootstrapAnalyzer(seed=args.seed, n_bootstrap=args.n_bootstrap)
    df['parameters'] = df.apply(analyzer.extract_parameters, axis=1)
    
    # Print detailed statistics about the data
    total_models = len(df)
    metadata_params = df.apply(lambda row: analyzer.extract_parameters_from_metadata(row['metadata']), axis=1).notna().sum()
    model_id_params = (df.apply(lambda row: analyzer.extract_parameters_from_metadata(row['metadata']), axis=1).isna() & 
                      df.apply(lambda row: analyzer.extract_parameters_from_model_id(row['model_id']), axis=1).notna()).sum()
    valid_models = df['parameters'].notna().sum()
    positive_models = df[df['parameters'] > 0]['parameters'].count()
    
    print(f"\nData Summary:")
    print(f"Total models: {total_models}")
    print(f"Parameters from metadata: {metadata_params}")
    print(f"Parameters from model_id: {model_id_params}")
    print(f"Models with valid parameter counts: {valid_models}")
    print(f"Models with positive parameter counts: {positive_models}")
    print(f"Models with missing parameter counts: {total_models - valid_models}")
    
    if positive_models == 0:
        print("Error: No valid positive parameter counts found in the data!")
        return
    
    # Show some examples of extracted parameters
    print(f"\nExample parameter extractions:")
    sample_df = df[df['parameters'].notna()].head(5)
    for idx, row in sample_df.iterrows():
        print(f"Model: {row['model_id'][:50]}...")
        print(f"Parameters: {row['parameters']:,.0f} ({row['parameters']:.2e})")
        print()
    
    # Run the analysis
    print("Running bootstrap analysis...")
    results = analyzer.bootstrap_analysis(df)
    
    # Save results to JSON
    results_file = os.path.join(output_dir, 'bootstrap_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Generate plots
    print("Generating plots...")
    analyzer.plot_bootstrap_results(df, results, output_dir)
    print(f"Plots saved to {output_dir}")
    
    # Print summary statistics
    print(f"\nSummary Statistics (with 95% CI):")
    param_dist = results['parameter_distribution']
    print(f"Mean parameters: {param_dist['mean']['mean']:,.0f} "
          f"[{param_dist['mean']['ci_lower']:,.0f}, {param_dist['mean']['ci_upper']:,.0f}]")
    print(f"Median parameters: {param_dist['median']['mean']:,.0f} "
          f"[{param_dist['median']['ci_lower']:,.0f}, {param_dist['median']['ci_upper']:,.0f}]")
    
    gap_analysis = results['gap_analysis']
    print(f"\nGap Analysis (20B-70B range):")
    print(f"Models in gap: {gap_analysis['models_in_gap']['mean']:.1f} "
          f"[{gap_analysis['models_in_gap']['ci_lower']:.1f}, {gap_analysis['models_in_gap']['ci_upper']:.1f}]")
    print(f"Gap percentage: {gap_analysis['gap_percentage']['mean']:.1f}% "
          f"[{gap_analysis['gap_percentage']['ci_lower']:.1f}%, {gap_analysis['gap_percentage']['ci_upper']:.1f}%]")

if __name__ == "__main__":
    main()
