"""
Visualize Model Family Tree

This script creates a visualization of a model family tree using networkx and matplotlib.
The visualization shows relationships between models with node sizes based on downloads
and colors based on depth in the tree.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

def clean_model_id(model_id):
    """Clean model ID by removing URL prefixes."""
    if model_id.startswith('https://huggingface.co/'):
        return model_id.replace('https://huggingface.co/', '')
    return model_id

def extract_metadata(metadata_str):
    """Extract downloads and likes from metadata string."""
    try:
        metadata = json.loads(metadata_str.replace("'", '"'))
        return metadata.get('downloads', 0), metadata.get('likes', 0)
    except:
        return 0, 0

def create_model_graph(df):
    """Create a directed graph from the model data."""
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for _, row in df.iterrows():
        model_id = clean_model_id(row['model_id'])
        downloads, likes = extract_metadata(row['metadata'])
        G.add_node(model_id, 
                  downloads=downloads,
                  likes=likes,
                  depth=row['depth'])
    
    # Add edges for parent-child relationships
    for _, row in df.iterrows():
        model_id = clean_model_id(row['model_id'])
        if pd.notna(row['children']):
            try:
                children = json.loads(row['children'].replace("'", '"'))
                for child in children:
                    child_id = clean_model_id(child)
                    if child_id in G:
                        G.add_edge(model_id, child_id)
            except:
                continue
    
    return G

def get_tree_layout(G):
    """Get a tree layout for the graph, handling disconnected components."""
    # Find all root nodes (nodes with no incoming edges)
    roots = [n for n, d in G.in_degree() if d == 0]
    if not roots:
        roots = [list(G.nodes())[0]]  # Use first node if no roots found
    
    # Sort roots by number of descendants
    roots.sort(key=lambda r: len(nx.descendants(G, r)), reverse=True)
    
    # Initialize positions dictionary
    pos = {}
    
    # Group nodes by depth
    depth_groups = {}
    for node in G.nodes():
        depth = G.nodes[node]['depth']
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(node)
    
    # Calculate positions for each depth level
    max_depth = max(depth_groups.keys()) if depth_groups else 0
    for depth, nodes in depth_groups.items():
        # Sort nodes by number of descendants to prioritize important nodes
        nodes.sort(key=lambda n: len(nx.descendants(G, n)), reverse=True)
        
        # Calculate horizontal positions
        n_nodes = len(nodes)
        if n_nodes > 0:
            # Calculate total width needed for this depth level
            total_width = n_nodes * 1.0  # 1.0 units per node
            
            # Calculate starting x position to center the group
            start_x = -total_width / 2
            
            # Assign positions to nodes
            for i, node in enumerate(nodes):
                x = start_x + (i + 0.5) * (total_width / n_nodes)
                y = -depth * 0.5  # Scale depth for better vertical spacing
                pos[node] = (x, y)
    
    return pos

def visualize_tree(G, output_file='phi2-tree-apr23.png'):
    """Create a visualization of the model tree."""
    plt.figure(figsize=(20, 12))
    
    # Calculate node sizes based on downloads (with a minimum size)
    downloads = [G.nodes[node]['downloads'] for node in G.nodes()]
    max_downloads = max(downloads) if downloads else 1
    node_sizes = [max(1000, (G.nodes[node]['downloads'] / max_downloads) * 5000 if max_downloads > 0 else 1000) for node in G.nodes()]
    
    # Get node colors based on depth
    node_colors = [G.nodes[node]['depth'] for node in G.nodes()]
    
    # Create labels with model name, likes, and downloads
    labels = {}
    for node in G.nodes():
        downloads = G.nodes[node]['downloads']
        likes = G.nodes[node]['likes']
        # Format large numbers with K/M suffix
        downloads_str = f"{downloads/1000:.1f}K" if downloads >= 1000 else str(downloads)
        downloads_str = f"{downloads/1000000:.1f}M" if downloads >= 1000000 else downloads_str
        labels[node] = f"{node.split('/')[-1]}\nL{likes} D{downloads_str}"
    
    # Get tree layout
    pos = get_tree_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_size=node_sizes,
                          node_color=node_colors,
                          cmap=plt.cm.viridis)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos,
                           labels=labels,
                           font_size=8,
                           font_weight='bold')
    
    # Draw edges with arrows
    ax = plt.gca()
    for (u, v) in G.edges():
        # Get positions
        pos_u = pos[u]
        pos_v = pos[v]
        
        # Create arrow
        arrow = plt.arrow(pos_u[0], pos_u[1],
                         pos_v[0] - pos_u[0], pos_v[1] - pos_u[1],
                         head_width=0.1,
                         head_length=0.2,
                         fc='gray',
                         ec='gray',
                         width=0.02,
                         length_includes_head=True)
        ax.add_patch(arrow)
    
    plt.title("Model Family Tree Visualization\nNode size based on downloads, color based on depth", 
              fontsize=16, pad=20)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = 'phi-2_finetunes_20250422_224221.csv'
        output_file = 'phi2-tree-apr23.png'
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Create the graph
    G = create_model_graph(df)
    
    # Visualize the tree
    visualize_tree(G, output_file)
    print(f"Tree visualization saved to {output_file}")

if __name__ == "__main__":
    main() 