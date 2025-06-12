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
from typing import Dict, List, Tuple, Any, Optional
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelAnalyzerConfig:
    """Configuration class for model analyzer"""
    
    PARAM_BINS = [0, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, float('inf')]
    PARAM_LABELS = ['0-1M', '1M-10M', '10M-100M', '100M-1B', '1B-10B', '10B-100B', '100B-1T', '>1T']
    
    PRECISION_PATTERNS = ['bf16', 'f16', 'fp16', 'f32', 'fp32', 'float16', 'float32']
    
    PARAM_KEYS = ['parameters', 'num_parameters', 'total_parameters', 'n_parameters', 'size', 'model_size']
    
    MULTIPLIERS = {
        'k': 1e3, 'thousand': 1e3,
        'm': 1e6, 'million': 1e6,
        'b': 1e9, 'billion': 1e9,
        'g': 1e9, 'giga': 1e9
    }

class ParameterExtractor:
    """Handles parameter extraction from various sources"""
    
    def __init__(self):
        self.config = ModelAnalyzerConfig()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.nxm_pattern = re.compile(r'(\d+)x(\d+)([bkmg])?$', re.IGNORECASE)
        self.suffix_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(thousand|million|billion|k|m|b|g)?$', re.IGNORECASE)
        self.param_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in [
                r'parameters:\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)',
                r'Total parameters:\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)',
                r'~?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)\s*parameters?',
                r'Params:\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)',
                r'[A-Za-z0-9_-]+-(\d+(?:\.\d+)?[BKMkmb])\b',
                r'(\d+x\d+[BKMkmb]?)',
                r'"num_parameters":\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?[BKMkmb]?)',
            ]
        ]
    
    def parse_number(self, text: Any) -> float:
        """Parse number strings with K/M/B/G suffixes into float values"""
        if pd.isna(text):
            return np.nan
        
        try:
            text_lower = str(text).strip().replace(',', '').lower()
            
            # Handle NxM format
            match = self.nxm_pattern.match(text_lower)
            if match:
                n1, n2, suffix = match.groups()
                multiplier = self.config.MULTIPLIERS.get(suffix, 1.0) if suffix else 1.0
                return float(n1) * float(n2) * multiplier
            
            # Handle suffixed numbers
            match = self.suffix_pattern.match(text_lower)
            if match:
                number_part, suffix = match.groups()
                multiplier = self.config.MULTIPLIERS.get(suffix, 1.0) if suffix else 1.0
                return float(number_part) * multiplier
            
            # Handle scientific notation and plain numbers
            return float(text_lower)
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse number '{text}': {e}")
            return np.nan
    
    def extract_from_metadata(self, metadata: Any) -> Optional[float]:
        """Extract parameters from metadata with robust error handling"""
        if pd.isna(metadata):
            return None
        
        # Parse metadata if it's a string
        if isinstance(metadata, str):
            metadata = self._parse_metadata_string(metadata)
        
        if not isinstance(metadata, dict):
            return None
        
        # Try standard parameter keys
        for key in self.config.PARAM_KEYS:
            if key in metadata:
                value = metadata[key]
                parsed = self._extract_from_value(value)
                if parsed is not None:
                    return parsed
        
        # Try safetensors structure
        safetensors = metadata.get('safetensors', {})
        if isinstance(safetensors, dict):
            # Check total first
            if 'total' in safetensors:
                parsed = self.parse_number(safetensors['total'])
                if not np.isnan(parsed):
                    return parsed
            
            # Check parameters by type
            params = safetensors.get('parameters', {})
            if isinstance(params, dict):
                for param_type in ['F32', 'BF16', 'I64']:
                    if param_type in params:
                        parsed = self.parse_number(params[param_type])
                        if not np.isnan(parsed):
                            return parsed
        
        # Recursive search in nested dictionaries
        return self._recursive_search(metadata)
    
    def _parse_metadata_string(self, metadata_str: str) -> dict:
        """Parse metadata string with multiple fallback strategies"""
        cleaned = metadata_str.replace('""', '"').replace("'", '"').strip()
        # Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r',([\s]*[}\]])', r'\1', cleaned)
        # Try ast.literal_eval first
        try:
            return ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            pass
        # Try json.loads
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse metadata string (malformed): {metadata_str[:100]}...")
            return {}
        # If parsed but no parameter info, do not warn
        return {}
    
    def _extract_from_value(self, value: Any) -> Optional[float]:
        """Extract parameter count from a value of unknown type"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            parsed = self.parse_number(value)
            return parsed if not np.isnan(parsed) else None
        elif isinstance(value, dict):
            # Handle nested parameter dictionaries
            for v in value.values():
                parsed = self._extract_from_value(v)
                if parsed is not None:
                    return parsed
        return None
    
    def _recursive_search(self, metadata: dict) -> Optional[float]:
        """Recursively search nested dictionaries for parameter information"""
        for value in metadata.values():
            if isinstance(value, dict):
                result = self.extract_from_metadata(value)
                if result is not None:
                    return result
        return None
    
    def extract_from_card(self, card_text: str) -> Optional[float]:
        """Extract parameters from model card text"""
        if pd.isna(card_text):
            return None
        
        card_str = str(card_text)
        
        # Try each compiled pattern
        for pattern in self.param_patterns:
            match = pattern.search(card_str)
            if match:
                parsed = self.parse_number(match.group(1))
                if not np.isnan(parsed):
                    return parsed
        
        return None
    
    def extract_from_model_id(self, model_id: str) -> Optional[float]:
        """Extract parameters from model ID"""
        if pd.isna(model_id):
            return None
        
        model_id_lower = str(model_id).lower()
        
        # Try standard suffix pattern
        match = re.search(r'(\d+(?:\.\d+)?)([bkmg])', model_id_lower)
        if match:
            num, suffix = match.groups()
            multiplier = self.config.MULTIPLIERS.get(suffix, 1.0)
            return float(num) * multiplier
        
        # Try NxM pattern
        match = self.nxm_pattern.match(model_id_lower)
        if match:
            n1, n2, suffix = match.groups()
            multiplier = self.config.MULTIPLIERS.get(suffix, 1.0) if suffix else 1.0
            return float(n1) * float(n2) * multiplier
        
        return None
    
    def extract_parameters(self, row: pd.Series) -> float:
        """Main parameter extraction method with fallback strategy"""
        # Try metadata first (most reliable)
        result = self.extract_from_metadata(row.get('metadata'))
        if result is not None:
            return result
        
        # Try model card
        result = self.extract_from_card(row.get('card'))
        if result is not None:
            return result
        
        # Try model ID as last resort
        result = self.extract_from_model_id(row.get('model_id'))
        if result is not None:
            return result
        
        return np.nan

class ModelAnalyzer:
    """Main model analysis class"""
    
    def __init__(self, config: Optional[ModelAnalyzerConfig] = None):
        self.config = config or ModelAnalyzerConfig()
        self.extractor = ParameterExtractor()
    
    def analyze_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze parameters in the dataset"""
        logger.info("Extracting parameters from dataset...")
        
        # Extract parameters
        df['parameters'] = df.apply(self.extractor.extract_parameters, axis=1)
        
        # Extract precision
        df['precision'] = df.apply(self.extract_precision, axis=1)
        
        # Create parameter size bins
        df['parameter_size'] = self._create_parameter_bins(df['parameters'])
        
        # Add parameter source information
        df['parameter_source'] = df.apply(self._identify_parameter_source, axis=1)
        
        logger.info(f"Successfully extracted parameters for {df['parameters'].notna().sum()} models")
        return df
    
    def _create_parameter_bins(self, parameters: pd.Series) -> pd.Series:
        """Create parameter size bins"""
        return pd.cut(parameters, bins=self.config.PARAM_BINS, labels=self.config.PARAM_LABELS)
    
    def _identify_parameter_source(self, row: pd.Series) -> str:
        """Identify the source of parameter information"""
        # Check each source in order of preference
        if self.extractor.extract_from_metadata(row.get('metadata')) is not None:
            return 'metadata'
        elif self.extractor.extract_from_card(row.get('card')) is not None:
            return 'card'
        elif self.extractor.extract_from_model_id(row.get('model_id')) is not None:
            return 'model_id'
        else:
            return 'unknown'
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistics"""
        stats = {
            'total_models': len(df),
            'models_with_parameters': df['parameters'].notna().sum(),
            'parameter_coverage': df['parameters'].notna().sum() / len(df) * 100,
            'parameter_stats': {
                'mean': df['parameters'].mean(),
                'median': df['parameters'].median(),
                'std': df['parameters'].std(),
                'min': df['parameters'].min(),
                'max': df['parameters'].max(),
                'q25': df['parameters'].quantile(0.25),
                'q75': df['parameters'].quantile(0.75)
            },
            'parameter_sources': df['parameter_source'].value_counts().to_dict(),
            'size_distribution': df['parameter_size'].value_counts().to_dict()
        }
        return stats
    
    def save_results(self, df: pd.DataFrame, output_dir: str, stats: Dict[str, Any]):
        """Save analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed dataset
        df.to_csv(os.path.join(output_dir, 'processed_models.csv'), index=False)
        
        # Save statistics
        with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir}")

    def extract_precision(self, row: pd.Series) -> str:
        """Extract floating point precision (BF16, F16, FP16, F32, FP32, float16, float32, etc.) from metadata, card, or model_id."""
        meta = row.get("metadata", None)
        if meta:
            if isinstance(meta, str):
                try:
                    meta = ast.literal_eval(meta)
                except Exception:
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = None
            if isinstance(meta, dict):
                # Check top-level keys
                for key in ["precision", "dtype"]:
                    val = meta.get(key)
                    if isinstance(val, str) and any(p in val.lower() for p in self.config.PRECISION_PATTERNS):
                        return val.upper()
                # Check safetensors/parameters
                st = meta.get("safetensors")
                if isinstance(st, dict):
                    params = st.get("parameters")
                    if isinstance(params, dict):
                        for k in params.keys():
                            if k.upper() in [p.upper() for p in self.config.PRECISION_PATTERNS]:
                                return k.upper()
                # Recursively check nested dicts
                for v in meta.values():
                    if isinstance(v, dict):
                        for k in v.keys():
                            if k.upper() in [p.upper() for p in self.config.PRECISION_PATTERNS]:
                                return k.upper()
        # 2. Check card text
        card = row.get("card", "")
        if isinstance(card, str):
            match = re.search(r"(bf16|f16|fp16|f32|fp32|float16|float32)", card, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        # 3. Check model_id
        model_id = row.get("model_id", "")
        if isinstance(model_id, str):
            match = re.search(r"(bf16|f16|fp16|f32|fp32|float16|float32)", model_id, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return "Unknown"

def save_precision_reference_table(output_dir):
    table_md = """# Floating Point Precision Reference\n\n| Precision | Memory Use | Speed | Stability | How to Enable                  | Hardware Support         |\n|-----------|------------|-------|-----------|-------------------------------|-------------------------|\n| FP32      | High       | Slow  | Best      | Default                       | All                     |\n| FP16      | Low        | Fast  | Lower     | torch_dtype=torch.float16      | Tensor Core GPUs        |\n| BF16      | Low        | Fast  | Higher    | TrainingArguments(bf16=True)   | Ampere+ GPUs            |\n| TF32      | Medium     | Fast  | High      | Default (Ampere+)              | Ampere+ GPUs (NVIDIA)   |\n"""
    with open(os.path.join(output_dir, "precision_reference.md"), "w") as f:
        f.write(table_md)

# Example usage
def main():
    """Run the model parameter and precision analysis on the joined_models CSV from Downloads."""
    # Path to the user's joined_models CSV
    csv_path = os.path.expanduser("~/Downloads/joined_models_20250529_135015.csv")
    output_dir = os.path.join(os.path.dirname(csv_path), "model_analysis_output")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Using data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(df)} rows from the CSV file")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    analyzer = ModelAnalyzer()
    df_analyzed = analyzer.analyze_parameters(df)
    stats = analyzer.generate_statistics(df_analyzed)
    analyzer.save_results(df_analyzed, output_dir, stats)
    save_precision_reference_table(output_dir)
    logger.info(f"Analysis complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
