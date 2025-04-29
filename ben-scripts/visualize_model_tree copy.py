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
# import pydot and other needed packages
import pydot

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

def create_model_graph(model_data):
    """Create a directed graph from the model data."""
    G = nx.DiGraph()
    
    def add_model_to_graph(model):
        """Recursively add model and its children to the graph."""
        # Clean the model ID
        model_id = clean_model_id(model['model_id'])
        
        # Get metadata with default values
        metadata = model.get('metadata', {})
        downloads = metadata.get('downloads', 0)
        likes = metadata.get('likes', 0)
        depth = model.get('depth', 0)
        
        # Add the model as a node with its attributes
        G.add_node(model_id, 
                  downloads=downloads,
                  likes=likes,
                  depth=depth)
        
        # Add edges to children
        for child in model.get('children', []):
            child_id = clean_model_id(child['model_id'])
            G.add_edge(model_id, child_id)
            add_model_to_graph(child)  # Recursively add children
    
    # Start with the root model
    add_model_to_graph(model_data)
    return G

def visualize_tree(G, output_file='model_tree.png'):
    """Visualize the model tree with node sizes based on downloads and colors based on depth."""
    plt.figure(figsize=(20, 12))
    
    # Get node attributes
    depths = [G.nodes[node]['depth'] for node in G.nodes]
    downloads = [G.nodes[node]['downloads'] for node in G.nodes]
    likes = [G.nodes[node]['likes'] for node in G.nodes]
    
    # Calculate node sizes based on downloads (with a minimum  and maximum size)
    node_sizes = [max(100, min(1000, d/10)) for d in downloads]
    
    # Create a color map based on depth
    colors = plt.cm.viridis([d/max(depths) for d in depths])
    
    # Draw the graph using spring layout to make it more readable and minimize edge crossings
    # Use pydot to draw the graph more beautifully 
    pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    #pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos, 
            with_labels=False,  # Remove default labels
            node_size=node_sizes,
            node_color=colors,
            font_size=8,
            font_weight='bold',
            edge_color='gray',
            arrows=True,
            arrowsize=10)
    
    # Add node labels with downloads and likes
    labels = {node: f"D{G.nodes[node]['downloads']}\nL{G.nodes[node]['likes']}" 
             #node: f"{node}\nD{G.nodes[node]['downloads']}\nL{G.nodes[node]['likes']}" 
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