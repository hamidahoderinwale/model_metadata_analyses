import requests
import pandas as pd
from tqdm import tqdm
import time
import json
from typing import Dict, List, Any, Set
import os
from pathlib import Path

class HuggingFaceModelCollector:
    def __init__(self):
        self.base_url = "https://huggingface.co/api"
        self.headers = {"User-Agent": "HuggingFace-Model-Collector/1.0"}
        self.visited_models: Set[str] = set()
        
    def get_models(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch models from Hugging Face API with pagination."""
        all_models = []
        page = 1
        page_size = 100
        
        with tqdm(total=limit, desc="Fetching models") as pbar:
            while len(all_models) < limit:
                params = {
                    "p": page,
                    "size": page_size,
                    "sort": "downloads",
                    "direction": -1
                }
                
                try:
                    response = requests.get(
                        f"{self.base_url}/models",
                        headers=self.headers,
                        params=params
                    )
                    response.raise_for_status()
                    
                    models = response.json()
                    if not models:
                        break
                        
                    all_models.extend(models)
                    pbar.update(len(models))
                    
                    if len(models) < page_size:
                        break
                        
                    page += 1
                    time.sleep(2)  # Increased rate limiting to 2 seconds
                    
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching models: {e}")
                    break
                    
        return all_models[:limit]
    
    def process_model(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single model to extract relevant information."""
        return {
            "model_id": model.get("id", ""),
            "author": model.get("author", ""),
            "downloads": model.get("downloads", 0),
            "likes": model.get("likes", 0),
            "library_name": model.get("library_name", ""),
            "pipeline_tag": model.get("pipeline_tag", ""),
            "tags": ", ".join(model.get("tags", [])),
            "created_at": model.get("created_at", ""),
            "last_modified": model.get("last_modified", ""),
            "private": model.get("private", False),
            "config": json.dumps(model.get("config", {})),
            "siblings": len(model.get("siblings", [])),
            "fine_tuned_models": self._get_fine_tuned_models(model),
            "adapters": self._get_adapters(model)
        }
    
    def _get_fine_tuned_models(self, model: Dict[str, Any]) -> str:
        """Extract fine-tuned models information."""
        model_id = model.get("id", "")
        if not model_id:
            return ""
            
        try:
            # URL encode the model ID
            encoded_model_id = model_id.replace("/", "%2F")
            
            # Use the correct URL format for finding fine-tuned models
            search_url = f"{self.base_url}/models"
            params = {
                "other": f"base_model:finetune:{encoded_model_id}",
                "sort": "downloads",
                "direction": -1,
                "limit": 100  # Limit to 100 fine-tuned models per base model
            }
            
            response = requests.get(
                search_url,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            
            fine_tuned_models = response.json()
            # Extract just the model IDs
            model_ids = [m["id"] for m in fine_tuned_models]
            print(f"Found {len(model_ids)} fine-tuned models for {model_id}")
            return ", ".join(model_ids)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching fine-tuned models for {model_id}: {e}")
            return ""
    
    def _get_adapters(self, model: Dict[str, Any]) -> str:
        """Extract adapters information."""
        model_id = model.get("id", "")
        if not model_id:
            return ""
            
        # Search for models that are adapters of this model
        search_url = f"{self.base_url}/models"
        params = {
            "search": f"adapter_of:{model_id}",
            "sort": "downloads",
            "direction": -1,
            "limit": 100  # Limit to 100 adapters per model
        }
        
        try:
            response = requests.get(
                search_url,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            
            adapter_models = response.json()
            # Extract just the model IDs
            model_ids = [m["id"] for m in adapter_models]
            return ", ".join(model_ids)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching adapters for {model_id}: {e}")
            return ""
    
    def _get_model_details(self, model_id: str) -> Dict[str, Any]:
        """Fetch detailed information about a specific model."""
        try:
            response = requests.get(
                f"{self.base_url}/models/{model_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching model details for {model_id}: {e}")
            return {}
    
    def collect_model_tree(self, base_model_id: str, output_file: str = "model_tree.csv") -> None:
        """Recursively collect information about a model and all its descendants."""
        if base_model_id in self.visited_models:
            print(f"Skipping already visited model: {base_model_id}")
            return
            
        self.visited_models.add(base_model_id)
        print(f"\nProcessing model: {base_model_id}")
        
        # Get model details
        model_details = self._get_model_details(base_model_id)
        if not model_details:
            print(f"Could not fetch details for model: {base_model_id}")
            return
            
        # Process the model
        processed_model = self.process_model(model_details)
        
        # Get fine-tuned models
        fine_tuned_models_str = processed_model["fine_tuned_models"]
        fine_tuned_models = [m.strip() for m in fine_tuned_models_str.split(",") if m.strip()]
        
        print(f"Found {len(fine_tuned_models)} fine-tuned models for {base_model_id}")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df = pd.DataFrame([processed_model])
        file_exists = output_path.exists()
        df.to_csv(output_file, mode='a', header=not file_exists, index=False)
        print(f"Saved information for {base_model_id}")
        
        # Process fine-tuned models
        for fine_tuned_id in fine_tuned_models:
            print(f"Processing fine-tuned model: {fine_tuned_id}")
            self.collect_model_tree(fine_tuned_id, output_file)
            time.sleep(2)  # Rate limiting between API calls
    
    def collect_and_save(self, output_file: str = "huggingface_models.csv", limit: int = 1000):
        """Collect models and save to CSV."""
        print("Starting model collection...")
        models = self.get_models(limit)
        
        print("Processing models...")
        processed_models = []
        for model in tqdm(models, desc="Processing models"):
            processed_models.append(self.process_model(model))
        
        print("Saving to CSV...")
        df = pd.DataFrame(processed_models)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

if __name__ == "__main__":
    collector = HuggingFaceModelCollector()
    # Example usage of the recursive collection
    # collector.collect_model_tree("bert-base-uncased", "bert_model_tree.csv")
    collector.collect_and_save(limit=1000)  # Adjust limit as needed 