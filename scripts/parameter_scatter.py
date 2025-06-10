import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import json
import ast
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def parse_number(text):
    """Parse number strings with K/M/B suffixes and NxM format into float values."""
    if pd.isna(text):
        return np.nan
    try:
        text_lower = str(text).strip().replace(',', '').lower()

        # Handle NxM format, e.g., "8x22b" -> 8 * 22e9
        match_nxm = re.match(r'(\d+)x(\d+)([bkm])?$', text_lower, re.IGNORECASE)
        if match_nxm:
            n1 = float(match_nxm.group(1))
            n2 = float(match_nxm.group(2))
            suffix = match_nxm.group(3)
            multiplier = 1.0
            if suffix:
                multipliers = {'k': 1e3, 'm': 1e6, 'b': 1e9}
                multiplier = multipliers.get(suffix, 1.0)
            return n1 * n2 * multiplier

        # Handle numbers with K/M/B suffixes, e.g., "127k", "1.5m", "70b"
        match_suffix = re.match(r'(\d+(?:\\.\\d+)?)([bkm])?$', text_lower, re.IGNORECASE)
        if match_suffix:
            number_part = float(match_suffix.group(1))
            suffix = match_suffix.group(2)
            if suffix:
                multipliers = {'k': 1e3, 'm': 1e6, 'b': 1e9}
                return number_part * multipliers.get(suffix, 1.0)
            else: # No suffix, just a number
                return number_part

        # Handle scientific notation and plain numbers
        if 'e' in text_lower:
            return float(text_lower)
        
        return float(text_lower)
    except Exception:
        return np.nan

def extract_parameters_from_card(card_text):
    """Extract parameter counts from unstructured card text using multiple regex patterns."""
    if pd.isna(card_text):
        return None

    param_patterns = [
        r'parameters:\s*{\s*"F32":\s*(\d+)\s*}',  # parameters: {"F32": 85804039}
        r'"parameters":\s*{\s*"F32":\s*(\d+)\s*}', # "parameters": {"F32": 85804039}
        r'parameters:\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)',  # parameters: 127k or 1,234,567
        r'Total parameters:\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)', # Total parameters: 127k
        r'~?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)\s*parameters?',  # ~127k parameters
        r'Params:\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)',  # Params: 127k
        r'Params[\s\t]+(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)',  # Params    127k
        r'[A-Za-z0-9_-]+-(\d+(?:\.\d+)?[BKMkmb])\b', # Captures parameters in 'model-7b' format
        r'(\d+x\d+[BKMkmb]?)', # Added: Captures NxM format like 8x22B
        r'#Params:\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)', # #Params: 127k
        r'(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)\s*params?',   # 127k params
        r'(\d+(?:\.\d+)?\s*(?:[BbMmKk]|billion|million|thousand)?)\s*parameters?', # 127k parameters
        r'"num_parameters":\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)', # "num_parameters": 127k
        r'"total_parameters":\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)', # "total_parameters": 127k
        r'"?(param_count|numParams|num_parameters|total_parameters)"?\s*[:=]\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)',
        r'(\d+(?:\.\d+)?\s*(?:[BbMmKk]|billion|million|thousand)?)\s*params?',
        r'\b(\d+(?:\.\d+)?[BKMkmb])\b' # Captures standalone numbers with B/M/K suffixes (e.g., "8B", "70M")
    ]
    for pattern in param_patterns:
        match = re.search(pattern, card_text, re.IGNORECASE)
        if match:
            return parse_number(match.group(1))
    return None

def extract_parameters_from_metadata(metadata):
    """Recursively search for parameter counts in nested metadata dictionaries."""
    if not isinstance(metadata, dict):
        return None

    # Common keys that might store parameter counts
    candidate_keys = [
        'parameters', 'num_parameters', 'total_parameters', 'n_parameters', 'size', 'model_size'
    ]
    # Try direct keys
    for key in candidate_keys:
        if key in metadata:
            value = metadata[key]
            if isinstance(value, (int, float)):
                return value
            elif isinstance(value, str):
                parsed = parse_number(value)
                if not np.isnan(parsed):
                    return parsed
            elif isinstance(value, dict):
                # Sometimes parameters are nested, e.g., {"F32": 85804039}
                for v in value.values():
                    parsed = parse_number(v)
                    if not np.isnan(parsed):
                        return parsed
    # Search nested dicts
    for v in metadata.values():
        if isinstance(v, dict):
            found = extract_parameters_from_metadata(v)
            if found is not None:
                return found
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
                # If all parsing fails, try to extract parameters directly using a more robust regex
                param_match = re.search(r'"parameters":\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)', cleaned_str, re.IGNORECASE)
                if param_match:
                    return pd.Series({'parameters': parse_number(param_match.group(1))})
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

def create_parameter_bins(parameters):
    """Create meaningful parameter size bins with exact ranges"""
    bins = [0, 1e6, 1e7, 1e8, 1e9, 20e9, 70e9, float('inf')]
    labels = [
        'Tiny (0-1M)',
        'Small (1M-10M)',
        'Medium (10M-100M)',
        'Large (100M-1B)',
        'Huge (1B-20B)',
        'Gap (20B-70B)',
        'Massive (>70B)'
    ]
    return pd.cut(parameters, bins=bins, labels=labels)

def generate_plotly_visualizations(df, output_dir):
    """Generate interactive Plotly visualizations for parameter analysis"""
    print("\nGenerating interactive visualizations...")
    
    # Create parameter size bins
    df['parameter_size'] = create_parameter_bins(df['parameters'])
    
    # 1. Interactive Scatter Plot with Parameter Sizes
    fig1 = px.scatter(
        df,
        x='depth',
        y='parameters',
        color='parameter_size',
        hover_data=['model_id', 'parameter_source'],
        log_y=True,
        title='Model Parameters vs Depth (Interactive)',
        labels={
            'depth': 'Model Depth',
            'parameters': 'Number of Parameters (log scale)',
            'parameter_size': 'Parameter Size Category'
        }
    )
    fig1.write_html(os.path.join(output_dir, 'parameter_scatter_interactive.html'))
    
    # 2. Parameter Distribution by Source
    fig2 = px.box(
        df,
        x='parameter_source',
        y='parameters',
        color='parameter_source',
        log_y=True,
        title='Parameter Distribution by Source',
        labels={
            'parameter_source': 'Parameter Source',
            'parameters': 'Number of Parameters (log scale)'
        }
    )
    fig2.write_html(os.path.join(output_dir, 'parameter_distribution_by_source.html'))
    
    # 3. Parameter Size Distribution
    size_counts = df['parameter_size'].value_counts().reset_index()
    size_counts.columns = ['parameter_size', 'count']
    fig3 = px.bar(
        size_counts,
        x='parameter_size',
        y='count',
        title='Distribution of Model Sizes',
        labels={
            'parameter_size': 'Parameter Size Category',
            'count': 'Number of Models'
        }
    )
    fig3.write_html(os.path.join(output_dir, 'parameter_size_distribution.html'))
    
    # 4. Combined Dashboard
    fig4 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Parameters vs Depth',
            'Parameter Distribution by Source',
            'Model Size Distribution',
            'Parameter Statistics'
        )
    )
    
    # Add scatter plot
    fig4.add_trace(
        go.Scatter(
            x=df['depth'],
            y=df['parameters'],
            mode='markers',
            marker=dict(
                color=df['parameters'],
                colorscale='Viridis',
                showscale=True
            ),
            name='Parameters vs Depth',
            text=df['model_id'],
            hoverinfo='text+x+y'
        ),
        row=1, col=1
    )
    
    # Add box plot
    for source in df['parameter_source'].unique():
        source_data = df[df['parameter_source'] == source]
        fig4.add_trace(
            go.Box(
                y=source_data['parameters'],
                name=source,
                boxpoints='all'
            ),
            row=1, col=2
        )
    
    # Add bar plot
    fig4.add_trace(
        go.Bar(
            x=size_counts['parameter_size'],
            y=size_counts['count'],
            name='Size Distribution'
        ),
        row=2, col=1
    )
    
    fig4.update_layout(
        height=1000,
        width=1200,
        title_text='Model Parameter Analysis Dashboard',
        showlegend=True
    )
    
    fig4.write_html(os.path.join(output_dir, 'parameter_analysis_dashboard.html'))
    
    print("Interactive visualizations saved to output directory")

def analyze_parameter_gap(df, output_dir):
    """Analyze and visualize the parameter gap phenomenon with enhanced metrics"""
    print("\nAnalyzing parameter gap with enhanced metrics...")
    
    # Create parameter gap bins
    df['parameter_gap_category'] = create_parameter_bins(df['parameters'])
    
    # Basic gap statistics
    gap_stats = {
        'total_models': len(df),
        'models_in_gap': len(df[df['parameter_gap_category'] == 'Gap (20B-70B)']),
        'models_below_gap': len(df[df['parameter_gap_category'] == 'Huge (1B-20B)']),
        'models_above_gap': len(df[df['parameter_gap_category'] == 'Massive (>70B)']),
        'gap_percentage': (len(df[df['parameter_gap_category'] == 'Gap (20B-70B)']) / len(df)) * 100
    }
    
    # Enhanced gap analysis
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        
        # Yearly distribution analysis
        yearly_stats = {}
        for year in df['year'].unique():
            year_data = df[df['year'] == year]
            yearly_stats[str(year)] = {
                'total_models': len(year_data),
                'models_in_gap': len(year_data[year_data['parameter_gap_category'] == 'Gap (20B-70B)']),
                'models_below_gap': len(year_data[year_data['parameter_gap_category'] == 'Huge (1B-20B)']),
                'models_above_gap': len(year_data[year_data['parameter_gap_category'] == 'Massive (>70B)']),
                'gap_percentage': (len(year_data[year_data['parameter_gap_category'] == 'Gap (20B-70B)']) / len(year_data)) * 100
            }
        gap_stats['yearly_analysis'] = yearly_stats
        
        # Trend analysis
        gap_stats['trend_analysis'] = {
            'gap_trend': {
                'increasing': any(yearly_stats[str(y)]['gap_percentage'] > yearly_stats[str(y-1)]['gap_percentage'] 
                                for y in range(min(df['year'])+1, max(df['year'])+1)),
                'decreasing': any(yearly_stats[str(y)]['gap_percentage'] < yearly_stats[str(y-1)]['gap_percentage'] 
                                for y in range(min(df['year'])+1, max(df['year'])+1))
            }
        }
    
    # Model type analysis (if available)
    if 'model_type' in df.columns:
        type_stats = {}
        for model_type in df['model_type'].unique():
            type_data = df[df['model_type'] == model_type]
            type_stats[model_type] = {
                'total_models': len(type_data),
                'models_in_gap': len(type_data[type_data['parameter_gap_category'] == 'Gap (20B-70B)']),
                'models_below_gap': len(type_data[type_data['parameter_gap_category'] == 'Huge (1B-20B)']),
                'models_above_gap': len(type_data[type_data['parameter_gap_category'] == 'Massive (>70B)']),
                'gap_percentage': (len(type_data[type_data['parameter_gap_category'] == 'Gap (20B-70B)']) / len(type_data)) * 100
            }
        gap_stats['model_type_analysis'] = type_stats
    
    # Create enhanced visualizations
    
    # 1. Parameter Distribution with Log-scaled Bins
    plt.figure(figsize=(20, 10))
    # Create log-scaled bins
    log_bins = np.logspace(np.log10(df['parameters'].min()), np.log10(df['parameters'].max()), 50)
    
    # Plot histogram with KDE
    sns.histplot(data=df, x='parameters', bins=log_bins, log_scale=True, alpha=0.7)
    sns.kdeplot(data=df, x='parameters', log_scale=True, color='red', linewidth=2)
    
    # Add vertical lines for key parameter thresholds
    plt.axvline(x=1e6, color='gray', linestyle='--', alpha=0.5, label='1M parameters')
    plt.axvline(x=1e9, color='gray', linestyle='--', alpha=0.5, label='1B parameters')
    plt.axvline(x=20e9, color='red', linestyle='--', alpha=0.5, label='20B parameters')
    plt.axvline(x=70e9, color='red', linestyle='--', alpha=0.5, label='70B parameters')
    
    plt.xscale('log')
    plt.title('Distribution of Model Parameters', fontsize=14, pad=20)
    plt.xlabel('Number of Parameters (log scale)', fontsize=12)
    plt.ylabel('Number of Models', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add detailed statistics
    stats_text = f"""
    Total Models: {gap_stats['total_models']}
    Models in Gap (20B-70B): {gap_stats['models_in_gap']} ({gap_stats['gap_percentage']:.1f}%)
    Models Below Gap (<20B): {gap_stats['models_below_gap']}
    Models Above Gap (>70B): {gap_stats['models_above_gap']}
    
    Parameter Statistics:
    Mean: {df['parameters'].mean():.2e}
    Median: {df['parameters'].median():.2e}
    Std Dev: {df['parameters'].std():.2e}
    Min: {df['parameters'].min():.2e}
    Max: {df['parameters'].max():.2e}
    """
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Date-Parameter Histogram
    if 'date' in df.columns:
        plt.figure(figsize=(15, 8))
        
        # Create 2D histogram with improved colormap
        plt.hist2d(df['date'], df['parameters'], 
                  bins=[50, log_bins],  # Use same log bins for parameters
                  cmap='viridis',
                  norm=plt.LogNorm())  # Log scale for color intensity
        
        # Add colorbar with better formatting
        cbar = plt.colorbar(label='Number of Models')
        cbar.ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        
        # Add horizontal lines for key parameter thresholds
        plt.axhline(y=1e6, color='gray', linestyle='--', alpha=0.5, label='1M parameters')
        plt.axhline(y=1e9, color='gray', linestyle='--', alpha=0.5, label='1B parameters')
        plt.axhline(y=20e9, color='red', linestyle='--', alpha=0.5, label='20B parameters')
        plt.axhline(y=70e9, color='red', linestyle='--', alpha=0.5, label='70B parameters')
        
        plt.yscale('log')
        plt.title('Distribution of Model Parameters Over Time', fontsize=14)
        plt.xlabel('Creation Date', fontsize=12)
        plt.ylabel('Number of Parameters (log scale)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add legend
        plt.legend(loc='upper right')
        
        # Add statistics annotation
        yearly_counts = df.groupby(df['date'].dt.year)['parameters'].count()
        stats_text = f"""
        Total Models: {len(df)}
        Models per Year:
        {yearly_counts.to_string()}
        """
        plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'date_parameter_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Yearly Gap Percentage
    plt.figure(figsize=(12, 6))
    years = sorted(yearly_stats.keys())
    gap_percentages = [yearly_stats[year]['gap_percentage'] for year in years]
    plt.plot(years, gap_percentages, marker='o', linestyle='-', linewidth=2)
    plt.title('Percentage of Models in Parameter Gap Over Time', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Percentage of Models in Gap (20B-70B)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'parameter_gap_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Model Type Distribution (if available)
    if 'model_type' in df.columns:
        plt.figure(figsize=(12, 6))
        type_data = pd.DataFrame(type_stats).T
        type_data['gap_percentage'].plot(kind='bar')
        plt.title('Parameter Gap Distribution by Model Type', fontsize=14)
        plt.xlabel('Model Type', fontsize=12)
        plt.ylabel('Percentage of Models in Gap (20B-70B)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_gap_by_type.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Parameter Range Analysis
    plt.figure(figsize=(12, 6))
    range_stats = df['parameter_gap_category'].value_counts()
    
    # Define the desired order of categories
    desired_order = [
        'Tiny (0-1M)',
        'Small (1M-10M)',
        'Medium (10M-100M)',
        'Gap (20B-70B)', 
        'Large (100M-1B)',
        'Huge (1B-20B)',
        'Massive (>70B)'
    ]
    
    # Reorder the categories according to the desired order
    range_stats = range_stats.reindex(desired_order)
    
    # Create bar plot with custom colors
    colors = ['#1f77b4'] * (len(range_stats) - 1) + ['#ff7f0e']  # Blue for normal bars, Orange for gap
    ax = range_stats.plot(kind='bar', color=colors)
    
    plt.title('Distribution of Models Across Parameter Ranges', fontsize=14)
    plt.xlabel('Parameter Range', fontsize=12)
    plt.ylabel('Number of Models', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add count labels on top of each bar
    for i, v in enumerate(range_stats):
        plt.text(i, v, f'{v:,}', ha='center', va='bottom')
    
    # Add percentage labels
    total = range_stats.sum()
    for i, v in enumerate(range_stats):
        percentage = (v / total) * 100
        plt.text(i, v/2, f'{percentage:.1f}%', ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_range_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return gap_stats

def save_parameter_gap_data(df, output_dir):
    """Save detailed parameter gap analysis data"""
    print("\nSaving detailed parameter gap analysis data...")
    
    # Create detailed gap analysis DataFrame
    gap_data = df.copy()
    gap_data['parameter_gap_category'] = create_parameter_bins(gap_data['parameters'])
    
    # Add additional gap analysis columns
    gap_data['is_in_gap'] = gap_data['parameter_gap_category'] == 'Gap (20B-70B)'
    gap_data['gap_distance'] = gap_data.apply(
        lambda x: min(abs(x['parameters'] - 20e9), abs(x['parameters'] - 70e9)) 
        if x['parameters'] is not None else None, 
        axis=1
    )
    
    # Save detailed gap analysis data
    gap_data_path = os.path.join(output_dir, 'parameter_gap_detailed_analysis.csv')
    gap_data.to_csv(gap_data_path, index=False)
    print(f"Detailed gap analysis data saved to: {gap_data_path}")
    
    return gap_data

def generate_advanced_stats(df, output_dir):
    """Generate advanced statistics and improved dataset."""
    print("\nGenerating advanced statistics and improved dataset...")

    enhanced_df = df.copy()
    enhanced_df['parameter_source'] = 'unknown'
    enhanced_df.loc[enhanced_df['parameters'].notna(), 'parameter_source'] = 'metadata'
    enhanced_df.loc[enhanced_df['card_parameters'].notna() & enhanced_df['parameters'].isna(), 'parameter_source'] = 'card'

    # Create parameter size bins
    enhanced_df['parameter_size'] = create_parameter_bins(enhanced_df['parameters'])
    
    # Calculate size distribution
    size_distribution = enhanced_df['parameter_size'].value_counts().to_dict()
    size_distribution_percent = (enhanced_df['parameter_size'].value_counts(normalize=True) * 100).to_dict()
    
    # Calculate source distribution within each size category
    size_source_distribution = {}
    for size in enhanced_df['parameter_size'].unique():
        size_data = enhanced_df[enhanced_df['parameter_size'] == size]
        size_source_distribution[str(size)] = {
            'total': len(size_data),
            'from_metadata': int((size_data['parameter_source'] == 'metadata').sum()),
            'from_card': int((size_data['parameter_source'] == 'card').sum()),
            'percentage': float(size_distribution_percent.get(size, 0))
        }

    # Calculate parameter ranges for each size category
    size_ranges = {}
    for size in enhanced_df['parameter_size'].unique():
        size_data = enhanced_df[enhanced_df['parameter_size'] == size]
        if not size_data.empty:
            size_ranges[str(size)] = {
                'min': float(size_data['parameters'].min()),
                'max': float(size_data['parameters'].max()),
                'mean': float(size_data['parameters'].mean()),
                'median': float(size_data['parameters'].median())
            }

    # Generate advanced statistics
    stats = {
        'parameter_analysis': {
            'total_models': len(df),
            'models_with_parameters': int(df['parameters'].notna().sum()),
            'parameter_sources': {
                'from_metadata': int((df['parameter_source'] == 'metadata').sum()),
                'from_card': int((df['parameter_source'] == 'card').sum())
            },
            'size_distribution': {
                'counts': size_distribution,
                'percentages': size_distribution_percent,
                'by_source': size_source_distribution
            },
            'size_ranges': size_ranges,
            'statistics': {
                'mean': float(df['parameters'].mean()) if df['parameters'].notna().any() else None,
                'median': float(df['parameters'].median()) if df['parameters'].notna().any() else None,
                'std': float(df['parameters'].std()) if df['parameters'].notna().any() else None,
                'min': float(df['parameters'].min()) if df['parameters'].notna().any() else None,
                'max': float(df['parameters'].max()) if df['parameters'].notna().any() else None,
                'q1': float(df['parameters'].quantile(0.25)) if df['parameters'].notna().any() else None,
                'q3': float(df['parameters'].quantile(0.75)) if df['parameters'].notna().any() else None
            }
        },
        'depth_analysis': {
            'mean_depth': float(df['depth'].mean()) if df['depth'].notna().any() else None,
            'median_depth': float(df['depth'].median()) if df['depth'].notna().any() else None,
            'depth_std': float(df['depth'].std()) if df['depth'].notna().any() else None,
            'min_depth': float(df['depth'].min()) if df['depth'].notna().any() else None,
            'max_depth': float(df['depth'].max()) if df['depth'].notna().any() else None
        },
        'correlation_analysis': {
            'depth_parameter_correlation': float(df[['depth', 'parameters']].corr().iloc[0,1]) if df['parameters'].notna().any() else None
        }
    }

    # Add parameter gap analysis
    gap_stats = analyze_parameter_gap(df, output_dir)
    stats['parameter_gap_analysis'] = gap_stats

    # Save detailed parameter gap data
    gap_data = save_parameter_gap_data(df, output_dir)
    
    # Add detailed gap statistics to the JSON
    stats['parameter_gap_analysis'].update({
        'detailed_statistics': {
            'gap_distance': {
                'mean': float(gap_data['gap_distance'].mean()) if gap_data['gap_distance'].notna().any() else None,
                'median': float(gap_data['gap_distance'].median()) if gap_data['gap_distance'].notna().any() else None,
                'min': float(gap_data['gap_distance'].min()) if gap_data['gap_distance'].notna().any() else None,
                'max': float(gap_data['gap_distance'].max()) if gap_data['gap_distance'].notna().any() else None
            },
            'parameter_ranges': {
                'below_gap': {
                    'mean': float(gap_data[gap_data['parameters'] < 20e9]['parameters'].mean()) if len(gap_data[gap_data['parameters'] < 20e9]) > 0 else None,
                    'median': float(gap_data[gap_data['parameters'] < 20e9]['parameters'].median()) if len(gap_data[gap_data['parameters'] < 20e9]) > 0 else None,
                    'count': int(len(gap_data[gap_data['parameters'] < 20e9]))
                },
                'in_gap': {
                    'mean': float(gap_data[gap_data['is_in_gap']]['parameters'].mean()) if gap_data['is_in_gap'].any() else None,
                    'median': float(gap_data[gap_data['is_in_gap']]['parameters'].median()) if gap_data['is_in_gap'].any() else None,
                    'count': int(gap_data['is_in_gap'].sum())
                },
                'above_gap': {
                    'mean': float(gap_data[gap_data['parameters'] > 70e9]['parameters'].mean()) if len(gap_data[gap_data['parameters'] > 70e9]) > 0 else None,
                    'median': float(gap_data[gap_data['parameters'] > 70e9]['parameters'].median()) if len(gap_data[gap_data['parameters'] > 70e9]) > 0 else None,
                    'count': int(len(gap_data[gap_data['parameters'] > 70e9]))
                }
            }
        }
    })

    # Save enhanced dataset
    enhanced_csv_path = os.path.join(output_dir, 'enhanced_model_parameters.csv')
    enhanced_df.to_csv(enhanced_csv_path, index=False)

    # Save advanced statistics
    advanced_stats_path = os.path.join(output_dir, 'advanced_analysis_stats.json')
    with open(advanced_stats_path, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"\nEnhanced dataset saved to: {enhanced_csv_path}")
    print(f"Advanced statistics saved to: {advanced_stats_path}")

    return stats

def plot_parameter_scatter(df, output_dir):
    """Generate scatter plot of model parameters."""
    print("\nExtracting parameters from metadata and cards...")
    
    # Extract parameters from metadata
    metadata_fields = df['metadata'].apply(extract_metadata_fields)
    df = pd.concat([df, metadata_fields], axis=1)
    
    # Extract parameters from card text
    df['card_parameters'] = df['card'].apply(extract_parameters_from_card)
    
    # Combine parameters from both sources, preferring metadata over card
    df['parameters'] = df['parameters'].fillna(df['card_parameters'])
    
    # Convert parameters to numeric
    df['parameters'] = pd.to_numeric(df['parameters'], errors='coerce')
    
    # Drop rows with invalid parameters
    df = df.dropna(subset=['parameters'])
    
    # Track parameter source
    df['parameter_source'] = np.where(
        df['parameters'].notna() & df['card_parameters'].isna(), 'metadata',
        np.where(df['parameters'].notna() & df['card_parameters'].notna(), 'card', 'unknown')
    )
    
    print(f"\nParameter extraction summary:")
    print(f"Total models: {len(df)}")
    print(f"Models with parameters from metadata: {(df['parameter_source'] == 'metadata').sum()}")
    print(f"Models with parameters from card: {(df['parameter_source'] == 'card').sum()}")
    
    # Generate advanced statistics and enhanced dataset
    advanced_stats = generate_advanced_stats(df, output_dir)
    
    # Generate Plotly visualizations
    generate_plotly_visualizations(df, output_dir)
    
    # Create matplotlib scatter plot
    plt.figure(figsize=(12, 8))
    
    # Create parameter size bins
    df['parameter_size'] = create_parameter_bins(df['parameters'])
    
    # Create a mask for models in the gap region
    gap_mask = (df['parameters'] >= 20e9) & (df['parameters'] <= 70e9)
    
    # Create a single scatter plot with different markers for gap models
    scatter = sns.scatterplot(
        data=df,
        x='depth',
        y='parameters',
        hue='parameter_size',
        style=gap_mask.map({True: 'Gap', False: 'Non-Gap'}),
        markers={'Gap': 's', 'Non-Gap': 'o'},
        s=gap_mask.map({True: 100, False: 60}),  # Larger size for gap models
        alpha=0.7,
        palette='viridis'
    )
    
    # Add shaded region for the gap
    plt.axhspan(20e9, 70e9, color='red', alpha=0.1, label='Parameter Gap (20B-70B)')
    
    # Add trend line if enough data points
    if df['parameters'].notna().sum() > 1:
        sns.regplot(data=df, x='depth', y='parameters', scatter=False, color='red', label='Trend Line')
    
    plt.yscale('log')
    plt.title('Model Parameters vs Depth (Colored by Parameter Size)', fontsize=14)
    plt.xlabel('Model Depth', fontsize=12)
    plt.ylabel('Number of Parameters (log scale)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient to plot
    correlation = advanced_stats['correlation_analysis']['depth_parameter_correlation']
    if correlation is not None:
        plt.annotate(
            f'Correlation: {correlation:.2f}',
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    # Update legend to be more concise
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate entries and combine parameter size and gap status
    unique_labels = []
    unique_handles = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    
    plt.legend(unique_handles, unique_labels, title='Parameter Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save plot
    output_path = os.path.join(output_dir, 'parameter_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {output_path}")

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"parameter_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Use the specific CSV file path
    input_path = '/Users/hamidahoderinwale/Desktop/joined_models_20250529_135015.csv'
    print(f"Using data from: {input_path}")

    try:
        # Load data
        df = pd.read_csv(input_path)
        print(f"Successfully loaded {len(df)} rows from the CSV file")

        # Generate plot and stats
        plot_parameter_scatter(df, output_dir)

    except FileNotFoundError as e:
        print(f"Error: Could not find the CSV file at {input_path}")
        print(f"Error details: {str(e)}")
    except Exception as e:
        print(f"An error occurred while processing the data: {str(e)}")

if __name__ == "__main__":
    main()
