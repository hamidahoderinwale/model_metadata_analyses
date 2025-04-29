"""
Script to visualize model family trees from Hugging Face Hub.
Supports both JSON and CSV input formats.
"""

import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import ast

def clean_model_id(model_id):
    """Clean model ID by removing URL prefixes."""
    if isinstance(model_id, str):
        if model_id.startswith('https://huggingface.co/'):
            return model_id.split('https://huggingface.co/')[-1]
    return model_id

def parse_metadata(metadata_str):
    """Parse metadata string into a dictionary."""
    try:
        if isinstance(metadata_str, str):
            return json.loads(metadata_str)
        return metadata_str
    except:
        return {}

def create_model_graph(data):
    """Create a directed graph from model data."""
    G = nx.DiGraph()
    
    # Handle both JSON and CSV data structures
    if isinstance(data, dict) and 'models' in data:
        models = data['models']
    else:
        models = data.to_dict('records')
    
    for model in models:
        model_id = clean_model_id(model['model_id'])
        metadata = parse_metadata(model['metadata'])
        
        # Add node with metadata
        G.add_node(model_id, 
                  #The depth is its own column in the csv
                  depth=model['depth'],
                  #downloads and likes are in the metadata column
                  downloads=metadata.get('downloads', 0),
                  likes=metadata.get('likes', 0))
        
        # Add edges for children
        children = model['children']
        if isinstance(children, str):
            try:
                children = ast.literal_eval(children)
            except:
                children = []
        
        for child in children:
            child_id = clean_model_id(child)
            G.add_edge(model_id, child_id)
    
    return G

def visualize_tree(G, output_file='model_tree.png'):
    """Visualize the model tree with node sizes based on downloads and colors based on depth."""
    plt.figure(figsize=(20, 12))
    
    # Get node attributes
    depths = [G.nodes[node]['depth'] for node in G.nodes]
    downloads = [G.nodes[node]['downloads'] for node in G.nodes]
    likes = [G.nodes[node]['likes'] for node in G.nodes]
    
    # Calculate node sizes based on downloads (with a minimum size)
    node_sizes = [max(100, d/10) for d in downloads]
    
    # Create a color map based on depth
    colors = plt.cm.viridis([d/max(depths) for d in depths])
    
    # Draw the graph
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos, 
            with_labels=True,
            node_size=node_sizes,
            node_color=colors,
            font_size=8,
            font_weight='bold',
            edge_color='gray',
            arrows=True,
            arrowsize=10)
    
    # Add node labels with downloads and likes
    labels = {node: f"{node}\nD{G.nodes[node]['downloads']}\nL{G.nodes[node]['likes']}" 
             for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=6)
    
    plt.title("Model Family Tree", fontsize=16)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tree visualization saved to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_model_tree.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'model_tree.png'
    
    # Determine file type and load data
    file_path = Path(input_file)
    if file_path.suffix.lower() == '.json':
        with open(input_file, 'r') as f:
            data = json.load(f)
    elif file_path.suffix.lower() == '.csv':
        data = pd.read_csv(input_file)
    else:
        print("Error: Input file must be either JSON or CSV")
        sys.exit(1)
    
    # Create and visualize the graph
    G = create_model_graph(data)
    visualize_tree(G, output_file)

if __name__ == "__main__":
    main() 