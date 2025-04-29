"""
Model Family Tree Collector

This script collects information about a model and its family tree (ancestors and descendants)
from the Hugging Face Hub and saves it to a CSV file. It uses the huggingface_hub library
to access model information and implements caching to reduce API calls.

Usage:
    python collect_model_family.py model_id [output_file]

Example:
    python collect_model_family.py "deepseek-ai/DeepSeek-V3" "deepseek_family_tree.csv"
"""

import sys
from huggingface_hub import ModelCard, HfApi, RepoCard
import requests
import pandas as pd
from collections import defaultdict
from pathlib import Path
import time
from typing import Set, Dict, List, Any
import json

class CachedModelCard(ModelCard):
    """
    A cached version of ModelCard that reduces API calls by storing previously fetched cards.
    """
    _cache = {}

    @classmethod
    def load(cls, model_id: str, **kwargs) -> "ModelCard":
        """
        Load a model card, using cache if available.
        
        Args:
            model_id: The ID of the model to load
            **kwargs: Additional arguments to pass to ModelCard.load
            
        Returns:
            The loaded ModelCard or None if loading failed
        """
        if model_id not in cls._cache:
            try:
                print('REQUEST ModelCard:', model_id)
                cls._cache[model_id] = super().load(model_id, **kwargs)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error loading model card for {model_id}: {e}")
                cls._cache[model_id] = None
        else:
            print('CACHED:', model_id)
        return cls._cache[model_id]

class ModelFamilyCollector:
    """
    Collects information about a model and its family tree from the Hugging Face Hub.
    """
    def __init__(self):
        self.api = HfApi()
        self.visited_models: Set[str] = set()
        self.model_info: List[Dict[str, Any]] = []
        
    def get_model_names_from_yaml(self, url: str) -> List[str]:
        """
        Get a list of parent model names from a YAML file.
        
        Args:
            url: The URL of the YAML file to fetch
            
        Returns:
            List of model names found in the YAML file
        """
        model_tags = []
        try:
            response = requests.get(url)
            if response.status_code == 200:
                model_tags.extend([item for item in response.content if '/' in str(item)])
        except Exception as e:
            print(f"Error fetching YAML from {url}: {e}")
        return model_tags

    def get_license_info(self, model: str) -> str:
        """
        Get the license information for a model.
        
        Args:
            model: The model ID
            
        Returns:
            One of: 'permissive', 'noncommercial', or 'unknown'
        """
        try:
            card = CachedModelCard.load(model)
            if card and hasattr(card.data, 'license'):
                license = card.data.license.lower()
                # Define permissive licenses
                permissive_licenses = ['mit', 'bsd', 'apache-2.0', 'openrail']
                if any(perm_license in license for perm_license in permissive_licenses):
                    return 'permissive'
                return 'noncommercial'
            return 'unknown'
        except Exception as e:
            print(f"Error retrieving license for {model}: {e}")
            return 'unknown'

    def process_model(self, model: str, parent_models: List[str]) -> Dict[str, Any]:
        """
        Process a single model and extract its information.
        
        Args:
            model: The model ID to process
            parent_models: List of parent model IDs
            
        Returns:
            Dictionary containing the model's information
        """
        try:
            card = CachedModelCard.load(model)
            if not card:
                return None
                
            card_dict = card.data.to_dict()
            
            # Get model details from the Hub API
            model_info = self.api.model_info(model)
            
            return {
                "model_id": model,
                "parent_models": ", ".join(parent_models),
                "author": card_dict.get('author', ''),
                "license": self.get_license_info(model),
                "downloads": getattr(model_info, 'downloads', 0),
                "likes": getattr(model_info, 'likes', 0),
                "library_name": card_dict.get('library_name', ''),
                "tags": ", ".join(card_dict.get('tags', [])),
                "created_at": str(getattr(model_info, 'created_at', '')),
                "last_modified": str(getattr(model_info, 'last_modified', '')),
                "pipeline_tag": card_dict.get('pipeline_tag', ''),
                "config": json.dumps(card_dict.get('config', {}))
            }
        except Exception as e:
            print(f"Error processing model {model}: {e}")
            return None

    def get_parent_models(self, model: str) -> List[str]:
        """
        Get the list of parent models for a given model.
        
        Args:
            model: The model ID to get parents for
            
        Returns:
            List of parent model IDs
        """
        try:
            card = CachedModelCard.load(model)
            if not card:
                return []
                
            card_dict = card.data.to_dict()
            model_tags = []
            
            # Check base_model field
            if 'base_model' in card_dict:
                model_tags = card_dict['base_model']
                if not isinstance(model_tags, list):
                    model_tags = [model_tags]
            
            # Check tags field
            if not model_tags and 'tags' in card_dict:
                tags = card_dict['tags']
                model_tags = [tag for tag in tags if '/' in tag]
            
            # Check YAML files
            if not model_tags:
                model_tags.extend(self.get_model_names_from_yaml(f"https://huggingface.co/{model}/blob/main/merge.yml"))
            if not model_tags:
                model_tags.extend(self.get_model_names_from_yaml(f"https://huggingface.co/{model}/blob/main/mergekit_config.yml"))
            
            return [tag for tag in model_tags if tag]  # Filter out empty tags
            
        except Exception as e:
            print(f"Error getting parent models for {model}: {e}")
            return []

    def collect_family_tree(self, model: str, parent_models: List[str] = None) -> None:
        """
        Recursively collect information about a model and its ancestors.
        
        Args:
            model: The model ID to start collecting from
            parent_models: List of parent model IDs (used internally for recursion)
        """
        if model in self.visited_models:
            print(f"Skipping already visited model: {model}")
            return
            
        print(f"\nProcessing model: {model}")
        self.visited_models.add(model)
        
        if parent_models is None:
            parent_models = []
        
        # Process current model
        model_info = self.process_model(model, parent_models)
        if model_info:
            self.model_info.append(model_info)
        
        # Get and process parent models
        parents = self.get_parent_models(model)
        print(f"Found {len(parents)} parent models for {model}")
        
        for parent in parents:
            self.collect_family_tree(parent, [model])
            time.sleep(1)  # Rate limiting

    def save_to_csv(self, output_file: str = "model_family_tree.csv") -> None:
        """
        Save the collected model information to a CSV file.
        
        Args:
            output_file: Path to save the CSV file
        """
        if not self.model_info:
            print("No model information to save.")
            return
            
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df = pd.DataFrame(self.model_info)
        df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        print(f"Total models collected: {len(self.model_info)}")

def main(model_id: str, output_file: str = None):
    """
    Main function to run the model family collection.
    
    Args:
        model_id: The ID of the model to start collecting from
        output_file: Optional path to save the CSV file
    """
    if not output_file:
        output_file = f"tree_info_{model_id.replace('/', '_')}.csv"
    
    collector = ModelFamilyCollector()
    collector.collect_family_tree(model_id)
    collector.save_to_csv(output_file)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        main(model_id, output_file)
    else:
        print("Please provide a model ID as an argument.")
        print("Usage: python collect_model_family.py model_id [output_file]") 