import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import warnings
import matplotlib.font_manager as fm

# Configure matplotlib with safe defaults
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 50  # Limit number of open figures

# Function to get a safe font
def get_safe_font():
    """Get a font that is safe for rendering"""
    # Try to find a font that supports basic characters
    safe_fonts = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    for font in safe_fonts:
        try:
            if font in fm.findSystemFonts():
                return font
        except:
            continue
    return 'sans-serif'

## number parsing script
def parse_number(text):
    """Parse number strings with K/M/B suffixes into float values"""
    try:
        text = text.replace(',', '').lower()
        if 'k' in text:
            return float(text.replace('k', '')) * 1e3
        elif 'm' in text:
            return float(text.replace('m', '')) * 1e6
        elif 'b' in text:
            return float(text.replace('b', '')) * 1e9
        else:
            return float(text)
    except Exception:
        return np.nan

def extract_from_card(card_text):
    """Extract parameter information from model card text"""
    if pd.isna(card_text):
        return None
    
    # Look for parameter information in various formats
    import re
    param_patterns = [
        r'parameters:\s*(\d+(?:\.\d+)?[BKM]?)',
        r'(\d+(?:\.\d+)?[BKM]?)\s*parameters',
        r'(\d+(?:\.\d+)?[BKM]?)\s*parameter',
        r'(\d+(?:\.\d+)?[BKM]?)\s*param'
    ]
    
    for pattern in param_patterns:
        match = re.search(pattern, card_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

# Function to clean text for plotting
def clean_text_for_plotting(text):
    """Clean text to handle CJK characters and other special characters"""
    if pd.isna(text):
        return 'Unknown'
    
    # First, handle special whitespace characters
    text = text.replace('\t', ' ')  # Replace tabs with spaces
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = text.replace('\r', ' ')  # Replace carriage returns with spaces
    text = ' '.join(text.split())  # Normalize all whitespace
    
    # Replace problematic characters with their ASCII equivalents or remove them
    replacements = {
        '・': '.',
        'ー': '-',
        '（': '(',
        '）': ')',
        '：': ':',
        '、': ',',
        '。': '.',
        '「': '"',
        '」': '"',
        '『': '"',
        '』': '"',
        '【': '[',
        '】': ']',
        '［': '[',
        '］': ']',
        '／': '/',
        '～': '~',
        '！': '!',
        '？': '?',
        '；': ';',
        '＆': '&',
        '＠': '@',
        '＃': '#',
        '＄': '$',
        '％': '%',
        '＾': '^',
        '＊': '*',
        '＋': '+',
        '＝': '=',
        '｜': '|',
        '＜': '<',
        '＞': '>',
        '｛': '{',
        '｝': '}',
        '｢': '[',
        '｣': ']',
        '､': ',',
        '･': '.',
        'ｰ': '-',
        'ﾞ': '"',
        'ﾟ': '"',
        '　': ' '  # Full-width space to half-width space
    }
    
    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove any remaining non-ASCII characters
    text = ''.join(char for char in text if ord(char) < 128)
    
    # Final whitespace cleanup
    text = ' '.join(text.split())
    
    return text

def setup_plot(figsize=(12, 6)):
    """Set up a plot with proper font handling and size"""
    fig, ax = plt.subplots(figsize=figsize)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return fig, ax

def safe_save_fig(fig, filename, dpi=100, bbox_inches='tight'):
    """Safely save a figure with error handling and size limits"""
    try:
        # First attempt with original settings
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    except Exception as e:
        print(f"Warning: Could not save {filename} with original settings. Trying alternative settings...")
        try:
            # Try with lower DPI and size limits
            fig.set_size_inches(12, 8)  # Cap maximum figure size
            fig.savefig(filename, dpi=min(dpi, 72), bbox_inches=bbox_inches)
        except Exception as e2:
            print(f"Error: Could not save {filename} even with alternative settings.")
            print(f"Original error: {str(e)}")
            print(f"Alternative error: {str(e2)}")

def create_output_dir():
    """Create timestamped output directory for plots with organized subdirectories"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"model_analysis_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    subdirs = {
        'plots': {
            'model_types': os.path.join(base_dir, 'plots', 'model_types'),
            'derivatives': os.path.join(base_dir, 'plots', 'derivatives'),
            'parameters': os.path.join(base_dir, 'plots', 'parameters'),
            'temporal': os.path.join(base_dir, 'plots', 'temporal'),
            'documentation': os.path.join(base_dir, 'plots', 'documentation')
        },
        'statistics': os.path.join(base_dir, 'statistics')
    }
    
    # Create all subdirectories
    for category in subdirs:
        if isinstance(subdirs[category], dict):
            for subdir in subdirs[category].values():
                os.makedirs(subdir, exist_ok=True)
        else:
            os.makedirs(subdirs[category], exist_ok=True)
    
    return subdirs

def load_and_preprocess_data(csv_path):
    """Load and preprocess the model data"""
    # Check if the path exists and is a file
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Safely convert string representations of lists to actual lists
    def safe_eval(x):
        try:
            if pd.isna(x) or x == "[]":
                return []
            # Clean the string to ensure it's a valid list representation
            x = x.strip()
            if not (x.startswith('[') and x.endswith(']')):
                return []
            # Remove any potential problematic characters
            x = x.replace("'", '"')
            return eval(x)
        except:
            return []
    
    # Convert string representations of lists to actual lists
    for col in ['children', 'adapters', 'quantized', 'merges']:
        if col in df.columns:
            df[col] = df[col].fillna("[]").apply(safe_eval)
    
    # Safely convert numeric columns
    def safe_numeric_convert(x):
        try:
            return pd.to_numeric(x, errors='coerce')
        except:
            return 0
    
    # Ensure numeric columns
    for col in ['children_count', 'adapters_count', 'quantized_count', 'merges_count', 'depth']:
        if col in df.columns:
            df[col] = df[col].apply(safe_numeric_convert).fillna(0).astype(int)
        else:
            # If column doesn't exist, create it with zeros
            df[col] = 0
    
    # Calculate total derivatives safely
    df['total_derivatives'] = (
        df['children'].apply(lambda x: len(x) if isinstance(x, list) else 0) + 
        df['adapters'].apply(lambda x: len(x) if isinstance(x, list) else 0) + 
        df['quantized'].apply(lambda x: len(x) if isinstance(x, list) else 0) + 
        df['merges'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    )
    
    # Add additional derived metrics
    df['has_derivatives'] = df['total_derivatives'] > 0
    df['derivative_diversity'] = (
        (df['children_count'] > 0).astype(int) +
        (df['adapters_count'] > 0).astype(int) +
        (df['quantized_count'] > 0).astype(int) +
        (df['merges_count'] > 0).astype(int)
    )
    
    # Print column information for debugging
    print("\nDataFrame columns:", df.columns.tolist())
    print("Number of rows:", len(df))
    print("\nSample of numeric columns:")
    numeric_cols = ['children_count', 'adapters_count', 'quantized_count', 'merges_count', 'depth', 'total_derivatives']
    print(df[numeric_cols].head())
    
    return df

def trim_labels(df, column, max_length=30):
    """Trim labels to maximum length and add ellipsis"""
    return df[column].astype(str).str.slice(0, max_length) + '...'

def limit_categories(df, column, n_categories=10):
    """Limit categories to top N and group rest as 'Other'"""
    top_categories = df[column].value_counts().nlargest(n_categories).index
    return df[column].apply(lambda x: x if x in top_categories else 'Other')

def plot_model_type_distribution_by_depth(df, output_dir):
    """Plot distribution of model types by depth with enhanced visualization"""
    # Extract and clean model types
    df['model_type'] = df['card'].apply(extract_model_type)
    df['model_type'] = df['model_type'].apply(clean_text_for_plotting)
    df['model_type'] = trim_labels(df, 'model_type')
    df['model_type'] = limit_categories(df, 'model_type')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    
    # Stacked bar plot
    type_by_depth = pd.crosstab(df['depth'], df['model_type'])
    type_by_depth.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_title("Model Type Distribution by Tree Depth", pad=20)
    ax1.set_xlabel("Tree Depth")
    ax1.set_ylabel("Number of Models")
    
    # Rotate x-tick labels
    ax1.tick_params(axis='x', rotation=45, ha='right')
    
    # Adjust legend position and size
    ax1.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    # Add percentage labels
    for c in ax1.containers:
        total = c.datavalues.sum()
        if total > 0:
            labels = [f'{v/total*100:.1f}%' if v > 0 else '' for v in c.datavalues]
            ax1.bar_label(c, labels=labels, label_type='center')
    
    # Pie chart for overall distribution
    overall_dist = df['model_type'].value_counts()
    ax2.pie(overall_dist, labels=overall_dist.index, autopct='%1.1f%%')
    ax2.set_title("Overall Model Type Distribution")
    
    # Adjust layout with more space for legend
    plt.subplots_adjust(right=0.85, hspace=0.3)
    
    # Save figure with error handling
    output_file = os.path.join(output_dir['plots']['model_types'], "model_type_by_depth.png")
    safe_save_fig(fig, output_file)
    plt.close()

def plot_derivative_correlation_by_depth(df, output_dir):
    """Plot correlation between different types of derivatives by depth"""
    depths = df['depth'].unique()
    fig, axes = plt.subplots(len(depths), 2, figsize=(12, 4*len(depths)))
    if len(depths) == 1:
        axes = axes.reshape(1, -1)
    
    for depth, (ax1, ax2) in zip(depths, axes):
        depth_df = df[df['depth'] == depth]
        
        # Correlation heatmap (limited to top correlations)
        correlation_matrix = depth_df[['children_count', 'adapters_count', 'quantized_count', 'merges_count']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax1)
        ax1.set_title(f"Correlation Between Model Derivatives at Depth {depth}")
        
        # Scatter matrix with size limits
        sns.scatterplot(data=depth_df, x='children_count', y='adapters_count', 
                       size='total_derivatives', hue='derivative_diversity',
                       sizes=(20, 200), ax=ax2)
        ax2.set_title(f"Relationship Between Fine-tunes and Adapters at Depth {depth}")
        ax2.set_xlabel("Number of Fine-tunes")
        ax2.set_ylabel("Number of Adapters")
        
        # Adjust legend position
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    plt.subplots_adjust(right=0.85, hspace=0.4)
    output_file = os.path.join(output_dir['plots']['derivatives'], "derivative_correlation_by_depth.png")
    safe_save_fig(fig, output_file)
    plt.close()

def plot_license_distribution_by_depth(df, output_dir):
    """Plot license distribution by model type and depth with enhanced visualization"""

    # Extract license information from either card or metadata
    def extract_license(card_text):
        if pd.isna(card_text):
            return 'unknown'

        import re
        license_patterns = [
            r'license:\s*([^\n]+)',
            r'license:\s*\[?([^\]]+)\]?',
            r'license type:\s*([^\n]+)'
        ]

        for pattern in license_patterns:
            match = re.search(pattern, card_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return 'unknown'

    # Extract licenses
    df['license'] = df['card'].apply(extract_license)

    # Clean and normalize license strings
    df['license'] = df['license'].fillna('unknown')
    df['license'] = df['license'].apply(clean_text_for_plotting)

    # Reduce long-tail licenses
    top_licenses = df['license'].value_counts().nlargest(10).index
    df['license'] = df['license'].apply(lambda x: x if x in top_licenses else 'Other')

    # Ensure CJK-safe font
    safe_font = get_safe_font()
    plt.rcParams['font.family'] = safe_font
    plt.rcParams['font.sans-serif'] = [safe_font]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])

    # Stacked bar plot by depth and type
    license_by_depth_type = pd.crosstab([df['depth'], df['model_type']], df['license'])
    license_by_depth_type.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_title("License Distribution by Tree Depth and Model Type", pad=20)
    ax1.set_xlabel("(Depth, Model Type)")
    ax1.set_ylabel("Count")
    ax1.legend(title="License Type", bbox_to_anchor=(1.05, 1))

    # Add percentage labels
    for c in ax1.containers:
        total = c.datavalues.sum()
        if total > 0:
            labels = [f'{v / total * 100:.1f}%' if v > 0 else '' for v in c.datavalues]
            ax1.bar_label(c, labels=labels, label_type='center')

    # Pie chart for overall license distribution
    overall_license = df['license'].value_counts()
    ax2.pie(overall_license, labels=overall_license.index, autopct='%1.1f%%')
    ax2.set_title("Overall License Distribution")

    # Save safely
    plt.tight_layout()
    output_file = os.path.join(output_dir['plots']['model_types'], "license_distribution_by_depth.png")
    safe_save_fig(fig, output_file)
    plt.close()


def plot_derivative_counts_by_depth(df, output_dir):
    """Plot distribution of derivative counts by depth with enhanced visualization"""
    derivative_cols = ['children_count', 'adapters_count', 'quantized_count', 'merges_count']
    
    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    
    for idx, col in enumerate(derivative_cols):
        # Box plot
        sns.boxplot(x='depth', y=col, data=df, ax=axes[idx])
        axes[idx].set_title(f"Distribution of {col.replace('_count', '')} by Tree Depth")
        axes[idx].set_xlabel("Tree Depth")
        axes[idx].set_ylabel("Number of Derivatives")
        
        # Add mean line
        means = df.groupby('depth')[col].mean()
        axes[idx].plot(range(len(means)), means, 'r--', label='Mean')
        axes[idx].legend()
        
        # Add count annotations
        for depth in df['depth'].unique():
            count = len(df[df['depth'] == depth])
            axes[idx].text(depth, axes[idx].get_ylim()[1], f'n={count}',
                         ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir['plots']['derivatives'], "derivative_counts_by_depth.png"), dpi=100, bbox_inches='tight')
    plt.close()

def plot_total_derivatives_by_depth(df, output_dir):
    """Plot distribution of total derivatives by depth with enhanced visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[2, 1])
    
    # Box plot with violin plot overlay
    sns.violinplot(x='depth', y='total_derivatives', data=df, ax=ax1, inner='box')
    ax1.set_title("Distribution of Total Derivatives by Tree Depth")
    ax1.set_xlabel("Tree Depth")
    ax1.set_ylabel("Total Number of Derivatives")
    
    # Add mean line and annotations
    means = df.groupby('depth')['total_derivatives'].mean()
    ax1.plot(range(len(means)), means, 'r--', label='Mean')
    for depth in df['depth'].unique():
        count = len(df[df['depth'] == depth])
        ax1.text(depth, ax1.get_ylim()[1], f'n={count}',
                ha='center', va='bottom')
    
    # Scatter plot with derivative diversity
    sns.scatterplot(data=df, x='depth', y='total_derivatives',
                   hue='derivative_diversity', size='derivative_diversity',
                   sizes=(20, 200), ax=ax2)
    ax2.set_title("Total Derivatives vs Depth with Derivative Diversity")
    ax2.set_xlabel("Tree Depth")
    ax2.set_ylabel("Total Number of Derivatives")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir['plots']['derivatives'], "total_derivatives_by_depth.png"), dpi=100, bbox_inches='tight')
    plt.close()

def plot_model_card_sizes_by_depth(df, output_dir):
    """Plot distribution of model card sizes by depth with enhanced visualization"""
    df['card_length'] = df['card'].str.len()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[2, 1])
    
    # Box plot with violin plot overlay
    sns.violinplot(x='depth', y='card_length', data=df, ax=ax1, inner='box')
    ax1.set_title("Model Card Size Distribution by Tree Depth")
    ax1.set_xlabel("Tree Depth")
    ax1.set_ylabel("Model Card Length (characters)")
    
    # Add mean line and annotations
    means = df.groupby('depth')['card_length'].mean()
    ax1.plot(range(len(means)), means, 'r--', label='Mean')
    for depth in df['depth'].unique():
        count = len(df[df['depth'] == depth])
        ax1.text(depth, ax1.get_ylim()[1], f'n={count}',
                ha='center', va='bottom')
    
    # Scatter plot with derivative count
    sns.scatterplot(data=df, x='depth', y='card_length',
                   hue='total_derivatives', size='total_derivatives',
                   sizes=(20, 200), ax=ax2)
    ax2.set_title("Model Card Size vs Depth with Total Derivatives")
    ax2.set_xlabel("Tree Depth")
    ax2.set_ylabel("Model Card Length (characters)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir['plots']['documentation'], "card_sizes_by_depth.png"), dpi=100, bbox_inches='tight')
    plt.close()

def plot_depth_distribution(df, output_dir):
    """Plot distribution of models across depths with enhanced visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[2, 1])
    
    # Bar plot with percentage labels
    depth_counts = df['depth'].value_counts().sort_index()
    bars = ax1.bar(depth_counts.index, depth_counts.values)
    ax1.set_title("Distribution of Models Across Tree Depths")
    ax1.set_xlabel("Tree Depth")
    ax1.set_ylabel("Number of Models")
    
    # Add percentage labels
    total = len(df)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height/total*100:.1f}%',
                ha='center', va='bottom')
    
    # Cumulative distribution
    cumulative = depth_counts.cumsum() / total * 100
    ax2.plot(cumulative.index, cumulative.values, 'b-', marker='o')
    ax2.set_title("Cumulative Distribution of Models by Depth")
    ax2.set_xlabel("Tree Depth")
    ax2.set_ylabel("Cumulative Percentage")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir['plots']['model_types'], "depth_distribution.png"), dpi=100, bbox_inches='tight')
    plt.close()

def plot_parameter_distribution_by_depth(df, output_dir):
    """Plot distribution of model parameters by depth with enhanced visualization"""
    # Extract parameters from model cards
    df['parameters'] = df['card'].apply(lambda c: parse_number(extract_from_card(c)))
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(10, 8))
    gs = plt.GridSpec(3, 2, figure=fig)
    
    # 1. Parameter distribution by depth (violin plot)
    ax1 = fig.add_subplot(gs[0, :])
    sns.violinplot(x='depth', y='parameters', data=df, ax=ax1, inner='box')
    ax1.set_title("Model Parameter Distribution by Tree Depth")
    ax1.set_xlabel("Tree Depth")
    ax1.set_ylabel("Number of Parameters")
    
    # Add mean line and annotations
    means = df.groupby('depth')['parameters'].mean()
    ax1.plot(range(len(means)), means, 'r--', label='Mean')
    for depth in df['depth'].unique():
        count = len(df[df['depth'] == depth])
        ax1.text(depth, ax1.get_ylim()[1], f'n={count}',
                ha='center', va='bottom')
    
    # 2. Parameter ranges by depth (stacked bar)
    ax2 = fig.add_subplot(gs[1, 0])
    parameter_ranges = pd.cut(df['parameters'], 
                            bins=[0, 1e6, 1e7, 1e8, 1e9, float('inf')],
                            labels=['<1M', '1M-10M', '10M-100M', '100M-1B', '>1B'])
    range_by_depth = pd.crosstab(df['depth'], parameter_ranges)
    range_by_depth.plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_title("Model Parameter Ranges by Tree Depth")
    ax2.set_xlabel("Tree Depth")
    ax2.set_ylabel("Number of Models")
    ax2.legend(title="Parameter Range", bbox_to_anchor=(1.05, 1))
    
    # Add percentage labels
    for c in ax2.containers:
        labels = [f'{v:.1f}%' if v > 0 else '' for v in c.datavalues / c.datavalues.sum() * 100]
        ax2.bar_label(c, labels=labels, label_type='center')
    
    # 3. Parameter vs Derivatives scatter
    ax3 = fig.add_subplot(gs[1, 1])
    sns.scatterplot(data=df, x='parameters', y='total_derivatives',
                   hue='depth', size='derivative_diversity',
                   sizes=(20, 200), ax=ax3)
    ax3.set_title("Parameters vs Total Derivatives")
    ax3.set_xlabel("Number of Parameters")
    ax3.set_ylabel("Total Number of Derivatives")
    
    # 4. Parameter trend by depth
    ax4 = fig.add_subplot(gs[2, :])
    sns.regplot(data=df, x='depth', y='parameters', ax=ax4, scatter_kws={'alpha':0.3})
    ax4.set_title("Parameter Size Trend by Tree Depth")
    ax4.set_xlabel("Tree Depth")
    ax4.set_ylabel("Number of Parameters")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir['plots']['parameters'], "parameter_analysis_by_depth.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    # Generate parameter statistics
    param_stats = {
        'overall': {
            'mean': df['parameters'].mean(),
            'median': df['parameters'].median(),
            'std': df['parameters'].std(),
            'min': df['parameters'].min(),
            'max': df['parameters'].max()
        },
        'by_depth': df.groupby('depth')['parameters'].agg(['mean', 'median', 'std', 'min', 'max']).to_dict(),
        'parameter_ranges': {
            'total': parameter_ranges.value_counts().to_dict(),
            'by_depth': range_by_depth.to_dict()
        }
    }
    
    # Save parameter statistics
    with open(os.path.join(output_dir['statistics'], "parameter_statistics.json"), 'w') as f:
        json.dump(param_stats, f, indent=4)

def plot_model_card_field_presence(df, output_dir):
    """Plot analysis of model card field presence by depth"""
    # Define the key model card fields to analyze
    model_card_fields = {
        'Model Details': [
            'model description',
            'developed by',
            'model type',
            'language',
            'license',
            'base model',
            'model sources',
            'library name',
            'pipeline tag'
        ],
        'Uses': [
            'intended uses',
            'direct use',
            'downstream use',
            'out of scope use',
            'limitations',
            'bias'
        ],
        'Training': [
            'training data',
            'preprocessing',
            'training procedure',
            'training hyperparameters',
            'framework versions',
            'mixed precision training'
        ],
        'Evaluation': [
            'evaluation',
            'testing data',
            'testing factors',
            'testing metrics',
            'results',
            'results summary'
        ],
        'Technical': [
            'model specs',
            'compute infrastructure',
            'hardware requirements',
            'software',
            'architecture',
            'tokenizer config'
        ],
        'Environmental': [
            'hardware type',
            'hours used',
            'cloud provider',
            'cloud region',
            'co2 emitted'
        ],
        'Documentation': [
            'citation',
            'citation bibtex',
            'citation apa',
            'glossary',
            'more information',
            'model card authors',
            'model card contact'
        ]
    }
    
    # Split categories into groups of 2 for better visualization
    categories = list(model_card_fields.keys())
    for i in range(0, len(categories), 2):
        group_categories = categories[i:i+2]
        
        # Create figure with subplots for this group
        fig, axes = plt.subplots(len(group_categories), 1, figsize=(12, 8))
        if len(group_categories) == 1:
            axes = [axes]
        
        for idx, category in enumerate(group_categories):
            fields = model_card_fields[category]
            # Calculate presence of fields in model cards
            presence_data = []
            for depth in sorted(df['depth'].unique()):
                depth_df = df[df['depth'] == depth]
                for field in fields:
                    field_presence = depth_df['card'].str.contains(field, case=False, na=False).mean() * 100
                    presence_data.append({
                        'Depth': depth,
                        'Field': clean_text_for_plotting(field),
                        'Presence': field_presence
                    })
            
            presence_df = pd.DataFrame(presence_data)
            
            # Create heatmap with limited data
            pivot_data = presence_df.pivot(index='Field', columns='Depth', values='Presence')
            # Limit to top 20 fields if there are too many
            if len(pivot_data) > 20:
                pivot_data = pivot_data.iloc[:20]
            
            sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[idx])
            axes[idx].set_title(f"{category} Fields Presence by Depth (%)")
            axes[idx].set_xlabel("Tree Depth")
            axes[idx].set_ylabel("Field")
            
            # Rotate x-tick labels
            axes[idx].tick_params(axis='x', rotation=45, ha='right')
        
        plt.subplots_adjust(right=0.95, hspace=0.4)
        output_file = os.path.join(output_dir['plots']['documentation'], f"model_card_field_presence_part{i//2+1}.png")
        safe_save_fig(fig, output_file)
        plt.close()
    
    # Create summary statistics for field presence
    field_stats = {}
    for category, fields in model_card_fields.items():
        category_presence = []
        for depth in sorted(df['depth'].unique()):
            depth_df = df[df['depth'] == depth]
            fields_present = sum(depth_df['card'].str.contains('|'.join(fields), case=False, na=False))
            total_fields = len(fields) * len(depth_df)
            category_presence.append({
                'Depth': depth,
                'Category': clean_text_for_plotting(category),
                'Presence': (fields_present / total_fields) * 100 if total_fields > 0 else 0
            })
        field_stats[category] = pd.DataFrame(category_presence)
    
    # Plot category presence by depth
    plt.figure(figsize=(12, 8))
    for category, stats in field_stats.items():
        plt.plot(stats['Depth'], stats['Presence'], marker='o', label=clean_text_for_plotting(category))
    
    plt.title("Model Card Category Completion by Tree Depth")
    plt.xlabel("Tree Depth")
    plt.ylabel("Field Presence (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.subplots_adjust(right=0.85)
    plt.savefig(os.path.join(output_dir['plots']['documentation'], "model_card_category_presence.png"), dpi=100, bbox_inches='tight')
    plt.close()

def generate_summary_stats(df, output_dir):
    """Generate enhanced summary statistics and save to JSON"""
    # Extract parameters from model cards
    def extract_parameters(card_text):
        if pd.isna(card_text):
            return None
        
        if 'parameters' in card_text.lower():
            import re
            param_patterns = [
                r'parameters:\s*(\d+(?:\.\d+)?[BKM]?)',
                r'(\d+(?:\.\d+)?[BKM]?)\s*parameters',
                r'(\d+(?:\.\d+)?[BKM]?)\s*parameter',
                r'(\d+(?:\.\d+)?[BKM]?)\s*param'
            ]
            
            for pattern in param_patterns:
                match = re.search(pattern, card_text, re.IGNORECASE)
                if match:
                    param_str = match.group(1)
                    if 'B' in param_str.upper():
                        return float(param_str.replace('B', '')) * 1e9
                    elif 'M' in param_str.upper():
                        return float(param_str.replace('M', '')) * 1e6
                    elif 'K' in param_str.upper():
                        return float(param_str.replace('K', '')) * 1e3
                    else:
                        return float(param_str)
        return None

    df['parameters'] = df['card'].apply(extract_parameters)
    
    # Define model card fields for analysis
    model_card_fields = {
        'Model Details': ['model_description', 'developers', 'funded_by', 'shared_by', 'model_type', 'language', 'license', 'base_model'],
        'Uses': ['direct_use', 'downstream_use', 'out_of_scope_use'],
        'Training': ['training_data', 'preprocessing', 'speeds_sizes_times'],
        'Evaluation': ['testing_data', 'testing_factors', 'testing_metrics', 'results', 'results_summary'],
        'Technical': ['model_specs', 'compute_infrastructure', 'hardware_requirements', 'software'],
        'Environmental': ['hardware_type', 'hours_used', 'cloud_provider', 'cloud_region', 'co2_emitted'],
        'Documentation': ['citation_bibtex', 'citation_apa', 'glossary', 'more_information', 'model_card_authors', 'model_card_contact']
    }
    
    # Calculate field presence statistics
    field_presence_stats = {}
    for category, fields in model_card_fields.items():
        category_stats = {}
        for depth in sorted(df['depth'].unique()):
            depth_df = df[df['depth'] == depth]
            field_stats = {}
            for field in fields:
                presence = depth_df['card'].str.contains(field, case=False, na=False).mean() * 100
                field_stats[field] = presence
            category_stats[depth] = field_stats
        field_presence_stats[category] = category_stats
    
    summary = {
        "total_models": len(df),
        "depth_distribution": df['depth'].value_counts().to_dict(),
        "model_types_by_depth": df.groupby('depth')['model_type'].value_counts().to_dict(),
        "total_derivatives_by_depth": df.groupby('depth')['total_derivatives'].agg(['mean', 'median', 'sum', 'std']).to_dict(),
        "license_distribution_by_depth": df.groupby('depth')['license'].value_counts().to_dict(),
        "card_length_by_depth": df.groupby('depth')['card_length'].agg(['mean', 'median', 'std']).to_dict(),
        "derivative_diversity_by_depth": df.groupby('depth')['derivative_diversity'].agg(['mean', 'median', 'std']).to_dict(),
        "parameters_by_depth": df.groupby('depth')['parameters'].agg(['mean', 'median', 'std', 'min', 'max']).to_dict(),
        "model_card_field_presence": field_presence_stats,
        "models_with_derivatives": {
            "total": df['has_derivatives'].sum(),
            "percentage": (df['has_derivatives'].sum() / len(df)) * 100,
            "by_depth": df.groupby('depth')['has_derivatives'].agg(['sum', 'mean']).to_dict()
        }
    }
    
    with open(os.path.join(output_dir['statistics'], "summary_stats.json"), 'w') as f:
        json.dump(summary, f, indent=4)

def plot_temporal_analysis_by_depth(df, output_dir):
    """Plot temporal analysis of models by depth with enhanced visualization"""
    # Extract last modified date from model cards
    def extract_last_modified(card_text):
        if pd.isna(card_text):
            return None
        
        import re
        date_patterns = [
            r'last modified:\s*(\d{4}-\d{2}-\d{2})',
            r'last_modified:\s*(\d{4}-\d{2}-\d{2})',
            r'last updated:\s*(\d{4}-\d{2}-\d{2})',
            r'last_updated:\s*(\d{4}-\d{2}-\d{2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, card_text, re.IGNORECASE)
            if match:
                return pd.to_datetime(match.group(1))
        
        return None

    # Extract dates and convert to datetime
    df['last_modified'] = df['card'].apply(extract_last_modified)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(10, 8))
    gs = plt.GridSpec(3, 2, figure=fig)
    
    # 1. Temporal distribution by depth
    ax1 = fig.add_subplot(gs[0, :])
    sns.boxplot(x='depth', y='last_modified', data=df, ax=ax1)
    ax1.set_title("Model Last Modified Date Distribution by Tree Depth")
    ax1.set_xlabel("Tree Depth")
    ax1.set_ylabel("Last Modified Date")
    plt.xticks(rotation=45)
    
    # Add count annotations
    for depth in df['depth'].unique():
        count = len(df[df['depth'] == depth])
        ax1.text(depth, ax1.get_ylim()[1], f'n={count}',
                ha='center', va='bottom')
    
    # 2. Temporal evolution of model types
    ax2 = fig.add_subplot(gs[1, 0])
    model_type_evolution = df.groupby(['last_modified', 'model_type']).size().unstack(fill_value=0)
    model_type_evolution.plot(kind='area', stacked=True, ax=ax2)
    ax2.set_title("Evolution of Model Types Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Number of Models")
    ax2.legend(title="Model Type", bbox_to_anchor=(1.05, 1))
    
    # 3. Temporal evolution of parameters
    ax3 = fig.add_subplot(gs[1, 1])
    sns.scatterplot(data=df, x='last_modified', y='parameters',
                   hue='depth', size='total_derivatives',
                   sizes=(20, 200), ax=ax3)
    ax3.set_title("Parameter Size Evolution Over Time")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Number of Parameters")
    
    # 4. Temporal evolution of derivatives
    ax4 = fig.add_subplot(gs[2, :])
    derivative_evolution = df.groupby('last_modified')['total_derivatives'].agg(['mean', 'median', 'std']).reset_index()
    ax4.plot(derivative_evolution['last_modified'], derivative_evolution['mean'], label='Mean')
    ax4.plot(derivative_evolution['last_modified'], derivative_evolution['median'], label='Median')
    ax4.fill_between(derivative_evolution['last_modified'],
                    derivative_evolution['mean'] - derivative_evolution['std'],
                    derivative_evolution['mean'] + derivative_evolution['std'],
                    alpha=0.2)
    ax4.set_title("Evolution of Model Derivatives Over Time")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Number of Derivatives")
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir['plots']['temporal'], "temporal_analysis_by_depth.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    # Generate temporal statistics
    temporal_stats = {
        'overall': {
            'earliest_date': df['last_modified'].min().strftime('%Y-%m-%d'),
            'latest_date': df['last_modified'].max().strftime('%Y-%m-%d'),
            'mean_date': df['last_modified'].mean().strftime('%Y-%m-%d'),
            'median_date': df['last_modified'].median().strftime('%Y-%m-%d')
        },
        'by_depth': df.groupby('depth')['last_modified'].agg([
            lambda x: x.min().strftime('%Y-%m-%d'),
            lambda x: x.max().strftime('%Y-%m-%d'),
            lambda x: x.mean().strftime('%Y-%m-%d'),
            lambda x: x.median().strftime('%Y-%m-%d')
        ]).to_dict(),
        'model_type_evolution': model_type_evolution.to_dict(),
        'derivative_evolution': derivative_evolution.to_dict()
    }
    
    # Save temporal statistics
    with open(os.path.join(output_dir['statistics'], "temporal_statistics.json"), 'w') as f:
        json.dump(temporal_stats, f, indent=4)

def plot_temporal_parameter_correlation(df, output_dir):
    """Plot correlation between temporal aspects and model parameters"""
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(10, 8))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Parameter size vs time by depth
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=df, x='last_modified', y='parameters',
                   hue='depth', size='total_derivatives',
                   sizes=(20, 200), ax=ax1)
    ax1.set_title("Parameter Size Evolution by Depth")
    ax1.set_xlabel("Last Modified Date")
    ax1.set_ylabel("Number of Parameters")
    
    # Add trend lines for each depth
    for depth in df['depth'].unique():
        depth_df = df[df['depth'] == depth]
        if len(depth_df) > 1:
            sns.regplot(data=depth_df, x='last_modified', y='parameters',
                       scatter=False, ax=ax1, label=f'Depth {depth}')
    
    # 2. Parameter ranges over time
    ax2 = fig.add_subplot(gs[0, 1])
    parameter_ranges = pd.cut(df['parameters'], 
                            bins=[0, 1e6, 1e7, 1e8, 1e9, float('inf')],
                            labels=['<1M', '1M-10M', '10M-100M', '100M-1B', '>1B'])
    range_evolution = df.groupby(['last_modified', parameter_ranges]).size().unstack(fill_value=0)
    range_evolution.plot(kind='area', stacked=True, ax=ax2)
    ax2.set_title("Evolution of Parameter Ranges Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Number of Models")
    ax2.legend(title="Parameter Range", bbox_to_anchor=(1.05, 1))
    
    # 3. Parameter vs derivatives over time
    ax3 = fig.add_subplot(gs[1, 0])
    sns.scatterplot(data=df, x='parameters', y='total_derivatives',
                   hue='last_modified', size='derivative_diversity',
                   sizes=(20, 200), ax=ax3)
    ax3.set_title("Parameters vs Derivatives Over Time")
    ax3.set_xlabel("Number of Parameters")
    ax3.set_ylabel("Total Number of Derivatives")
    
    # 4. Parameter growth rate by depth
    ax4 = fig.add_subplot(gs[1, 1])
    df['year'] = df['last_modified'].dt.year
    yearly_params = df.groupby(['year', 'depth'])['parameters'].mean().unstack()
    yearly_params.plot(kind='line', marker='o', ax=ax4)
    ax4.set_title("Yearly Average Parameter Size by Depth")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Average Number of Parameters")
    ax4.legend(title="Depth")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir['plots']['temporal'], "temporal_parameter_correlation.png"), dpi=100, bbox_inches='tight')
    plt.close()

def main():
    # Use the specified dataset path
    input_csv = '/Users/hamidahoderinwale/Desktop/HFAnalysisProj/model_metadata_analyses/scripts/joined_models_20250529_135015.csv'
    
    # Create output directory structure
    output_dir = create_output_dir()
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = load_and_preprocess_data(input_csv)
        
        # Generate all plots
        print("Generating plots...")
        try:
            plot_depth_distribution(df, output_dir)
            plot_model_type_distribution_by_depth(df, output_dir)
            plot_derivative_correlation_by_depth(df, output_dir)
            plot_license_distribution_by_depth(df, output_dir)
            plot_derivative_counts_by_depth(df, output_dir)
            plot_total_derivatives_by_depth(df, output_dir)
            plot_model_card_sizes_by_depth(df, output_dir)
            plot_parameter_distribution_by_depth(df, output_dir)
            plot_model_card_field_presence(df, output_dir)
            plot_temporal_analysis_by_depth(df, output_dir)
            plot_temporal_parameter_correlation(df, output_dir)
        except Exception as e:
            print(f"Warning: Some plots could not be generated: {str(e)}")
        
        # Generate summary statistics
        print("Generating summary statistics...")
        generate_summary_stats(df, output_dir)
        
        print(f"\nAnalysis complete! All outputs saved to: {output_dir['plots']['model_types'].split('/')[0]}/")
        print("\nDirectory structure:")
        print(f"- plots/")
        print(f"  - model_types/")
        print(f"  - derivatives/")
        print(f"  - parameters/")
        print(f"  - temporal/")
        print(f"  - documentation/")
        print(f"- statistics/")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure the CSV file exists at the specified path and has the correct extension (.csv)")
        print("Current path:", input_csv)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 
