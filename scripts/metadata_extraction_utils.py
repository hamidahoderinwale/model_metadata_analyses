import pandas as pd
import numpy as np
import json
import re
import logging
from datetime import datetime
from typing import Any, Optional, Tuple

# Set up logging (optional, can be configured in main script)
logger = logging.getLogger(__name__)
# Prevent propagation to the root logger if the main script configures it
# If not configured, this logger will use the default handler/config
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


def extract_parameters_from_metadata(metadata_str: Any) -> float:
    """
    Extract parameter count from metadata string.
    Handles NaN, non-string inputs, JSON/YAML formats, string patterns,
    and various parameter keys/suffixes.
    """
    if pd.isna(metadata_str):
        return np.nan

    if not isinstance(metadata_str, str):
        logger.debug(f"Metadata is not a string (type: {type(metadata_str)}): {metadata_str}")
        return np.nan

    # Clean the string for easier parsing
    cleaned_str = metadata_str.strip()

    # Try parsing as JSON
    try:
        metadata = json.loads(cleaned_str)
        if isinstance(metadata, dict):
            # Try common parameter keys in JSON
            param_keys = ['parameters', 'parameter_count', 'num_parameters', 'total_parameters', 'size', 'model_size']
            for key in param_keys:
                if key in metadata:
                    value = metadata[key]
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, str):
                        # Handle string values with K/M/B suffixes in JSON
                        value_lower = value.lower().replace(',', '').strip()
                        try:
                            if 'b' in value_lower:
                                return float(value_lower.replace('b', '')) * 1e9
                            elif 'm' in value_lower:
                                return float(value_lower.replace('m', '')) * 1e6
                            elif 'k' in value_lower:
                                return float(value_lower.replace('k', '')) * 1e3
                            else:
                                return float(value_lower)
                        except ValueError:
                            logger.debug(f"Could not parse string parameter value '{value}' from JSON.")
                            continue # Try next key or pattern

    except json.JSONDecodeError:
        logger.debug("Metadata is not valid JSON. Proceeding with regex/string parsing.")
        pass # Not JSON, continue to string/regex parsing

    # If not JSON or parameter not found in JSON, try string/regex parsing
    # Look for patterns like "7B parameters", "1.5B params", etc.
    patterns = [
        r'(\d+(?:\.\d+)?)\s*[Bb]\s*(?:parameters?|params?)',  # 7B parameters
        r'(\d+(?:\.\d+)?)\s*[Mm]\s*(?:parameters?|params?)',  # 500M parameters
        r'(\d+(?:\.\d+)?)\s*[Kk]\s*(?:parameters?|params?)',  # 100K parameters
        r'total\s*size\s*:\s*(\d+(?:\.\d+)?)\s*[Bb]', # total size : 7B
        r'total\s*size\s*:\s*(\d+(?:\.\d+)?)\s*[Mm]', # total size : 500M
        r'total\s*size\s*:\s*(\d+(?:\.\d+)?)\s*[Kk]', # total size : 100K
        r'total\s*:\s*(\d+(?:\.\d+)?)', # safetensors total: 123456
        r'(\d+(?:\.\d+)?)\s*(?:parameters?|params?)'          # plain numbers anywhere
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned_str.lower())
        if match:
            try:
                value = float(match.group(1))
                if 'b' in pattern.lower():
                    return value * 1e9
                elif 'm' in pattern.lower():
                    return value * 1e6
                elif 'k' in pattern.lower():
                    return value * 1e3
                else:
                    # Check if the pattern is one of the 'total size' or 'total' patterns
                    if 'total size' in pattern.lower() or 'total :' in pattern.lower():
                         # These patterns already imply the correct scale, no need for k/m/b check
                         return value
                    # For generic plain numbers, assume it's the raw count if no suffix
                    return value
            except ValueError:
                logger.debug(f"Could not convert regex match '{match.group(1)}' to float.")
                continue

    logger.debug(f"No parameter information found in metadata string: {metadata_str}")
    return np.nan

def extract_date_from_metadata(metadata_str: Any, date_type: str) -> Optional[datetime]:
    """
    Extracts created_at or last_modified date from metadata string.
    Handles NaN, non-string inputs, JSON/YAML formats, and various date keys.
    Returns a timezone-naive datetime object or None if extraction fails.
    """
    if pd.isna(metadata_str):
        return None

    if not isinstance(metadata_str, str):
        logger.debug(f"Metadata is not a string (type: {type(metadata_str)}): {metadata_str}")
        return None

    cleaned_str = metadata_str.strip()
    date_keys = {
        'created_at': ['created_at', 'model_created_at', 'createdAt', 'modelCreatedAt'],
        'last_modified': ['last_modified', 'model_last_modified', 'lastModified', 'modelLastModified']
    }

    # Try parsing as JSON
    try:
        metadata = json.loads(cleaned_str)
        if isinstance(metadata, dict):
            for key in date_keys.get(date_type, []):
                if key in metadata:
                    date_value = metadata[key]
                    if isinstance(date_value, str):
                        try:
                            # Parse date string, assume timezone-naive
                            dt_obj = datetime.fromisoformat(date_value.replace('Z', '+00:00')).replace(tzinfo=None)
                            return dt_obj
                        except ValueError:
                            logger.debug(f"Could not parse date string '{date_value}' for key '{key}' from JSON.")
                            continue # Try next key

    except json.JSONDecodeError:
        logger.debug("Metadata is not valid JSON. Proceeding with string parsing.")
        pass # Not JSON, continue to string parsing

    # If not JSON or date not found in JSON, try string parsing (less reliable)
    # Look for common date patterns in the string
    # This is a fallback and may not be accurate
    date_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?' # ISO 8601 like
    match = re.search(date_pattern, cleaned_str)
    if match:
        try:
            # Parse date string, assume timezone-naive
            dt_obj = datetime.fromisoformat(match.group(0).replace('Z', '+00:00')).replace(tzinfo=None)
            return dt_obj
        except ValueError:
            logger.debug(f"Could not parse date string '{match.group(0)}' from raw string.")
            pass

    logger.debug(f"No {date_type} information found in metadata string: {metadata_str}")
    return None


def extract_model_family(model_id: str, metadata_str: Any) -> Tuple[str, str]:
    """
    Extract model family using both model ID and metadata information.
    Returns a tuple of (family, confidence) where confidence is 'high' or 'low'.
    Handles NaN and non-string metadata inputs.
    """
    if pd.isna(model_id) or pd.isna(metadata_str) or not isinstance(metadata_str, str):
        return "Unknown", "low"

    cleaned_metadata_str = metadata_str.strip()

    # First try to get family from metadata (JSON)
    try:
        metadata = json.loads(cleaned_metadata_str)
        if isinstance(metadata, dict):
            # Check for explicit architecture information
            arch_keys = ['architecture', 'model_type', 'model_architecture', 'model_family', 'library_name']
            for key in arch_keys:
                if key in metadata:
                    arch = str(metadata[key]).lower()
                    # Map common architecture names/library names to families
                    if 'transformers' in arch: return "Transformer (Library)", "high" # Added library check
                    if 'bert' in arch: return "BERT", "high"
                    elif 'gpt' in arch: return "GPT", "high"
                    elif 't5' in arch: return "T5", "high"
                    elif 'roberta' in arch: return "RoBERTa", "high"
                    elif 'llama' in arch: return "LLaMA", "high"
                    elif 'mistral' in arch: return "Mistral", "high"
                    elif 'falcon' in arch: return "Falcon", "high"
                    elif 'bloom' in arch: return "BLOOM", "high"
                    elif 'opt' in arch: return "OPT", "high"
                    elif 'mpt' in arch: return "MPT", "high"
                    elif 'phi' in arch: return "Phi", "high"
                    elif 'gemma' in arch: return "Gemma", "high"
                    elif 'qwen' in arch: return "Qwen", "high"
                    elif 'yi' in arch: return "Yi", "high"
                    elif 'baichuan' in arch: return "Baichuan", "high"
                    elif 'chatglm' in arch: return "ChatGLM", "high"
                    elif 'internlm' in arch: return "InternLM", "high"
                    elif 'deepseek' in arch: return "DeepSeek", "high"
                    elif 'stablelm' in arch: return "StableLM", "high"
                    elif 'openllama' in arch: return "OpenLLaMA", "high"
                    elif 'pythia' in arch: return "Pythia", "high"
                    elif 'cerebras' in arch: return "Cerebras", "high"
                    elif 'mamba' in arch: return "Mamba", "high"
                    elif 'recurrent' in arch: return "Recurrent", "low"
                    elif 'rwkv' in arch: return "RWKV", "high"
                    elif 'transformer' in arch: return "Transformer", "low" # Generic transformer
    except json.JSONDecodeError:
        logger.debug("Metadata is not valid JSON for family extraction.")
        pass

    # If metadata doesn't help, try model ID
    model_id_lower = str(model_id).lower()

    # Define model family patterns with their confidence levels
    family_patterns = {
        'bert': ('BERT', 'high'),
        'gpt': ('GPT', 'high'),
        't5': ('T5', 'high'),
        'roberta': ('RoBERTa', 'high'),
        'llama': ('LLaMA', 'high'),
        'mistral': ('Mistral', 'high'),
        'falcon': ('Falcon', 'high'),
        'bloom': ('BLOOM', 'high'),
        'opt': ('OPT', 'high'),
        'gpt2': ('GPT-2', 'high'),
        'gpt-j': ('GPT-J', 'high'),
        'gpt-neo': ('GPT-Neo', 'high'),
        'gpt-neox': ('GPT-NeoX', 'high'),
        'mpt': ('MPT', 'high'),
        'phi': ('Phi', 'high'),
        'gemma': ('Gemma', 'high'),
        'qwen': ('Qwen', 'high'),
        'yi': ('Yi', 'high'),
        'baichuan': ('Baichuan', 'high'),
        'chatglm': ('ChatGLM', 'high'),
        'internlm': ('InternLM', 'high'),
        'deepseek': ('DeepSeek', 'high'),
        'stablelm': ('StableLM', 'high'),
        'openllama': ('OpenLLaMA', 'high'),
        'pythia': ('Pythia', 'high'),
        'cerebras': ('Cerebras', 'high'),
        'mamba': ('Mamba', 'high'),
        'recurrent': ('Recurrent', 'low'),
        'rwkv': ('RWKV', 'high'),
        'transformer': ('Transformer', 'low') # Generic transformer in ID
    }

    # Check for pattern matches in model ID
    for pattern, (family, confidence) in family_patterns.items():
        if pattern in model_id_lower:
            return family, confidence

    # Check for organization-specific patterns in model ID
    org_patterns = {
        'meta-ai': 'Meta',
        'google': 'Google',
        'microsoft': 'Microsoft',
        'huggingface': 'HuggingFace',
        'stabilityai': 'StabilityAI',
        'deepmind': 'DeepMind',
        'openai': 'OpenAI',
        'anthropic': 'Anthropic',
        'cohere': 'Cohere',
        'nvidia': 'NVIDIA',
        'amazon': 'Amazon',
        'apple': 'Apple',
        'salesforce': 'Salesforce',
        'facebook': 'Meta',
        'alibaba': 'Alibaba',
        'tencent': 'Tencent',
        'baidu': 'Baidu'
    }

    for org, family in org_patterns.items():
        if org in model_id_lower:
            return f"{family} Model", "low" # Label as low confidence if only org is matched

    return "Other", "low"


def extract_architecture_type(metadata_str: Any) -> str:
    """Extract architecture type from metadata."""
    if pd.isna(metadata_str) or not isinstance(metadata_str, str):
        return "Unknown"

    cleaned_str = metadata_str.strip()

    try:
        metadata = json.loads(cleaned_str)
        if isinstance(metadata, dict):
            # Check for architecture information
            arch_keys = ['architecture', 'model_type', 'model_architecture', 'library_name']
            for key in arch_keys:
                if key in metadata:
                    # Return the value directly for more detail
                    return str(metadata[key])
    except json.JSONDecodeError:
         logger.debug("Metadata is not valid JSON for architecture extraction.")
         pass # Not JSON, return Unknown for now

    return "Unknown"
